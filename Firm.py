import os

from Job import Job
from Machine import Machine
from gym import Env
from gym.spaces import Discrete, Box, Dict
from pathlib import Path
import numpy as np
import random
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.figure_factory as ff


class Firm(Env):

    def __init__(self, late_delivery=False, downtime=False, downtime_total=False, diminishing_value=1.0, scenario=None,
                 eval=False, wip=False):
        self.late_delivery = late_delivery
        self.downtime = downtime
        self.downtime_total = downtime_total
        self.wip = wip
        self.num_downtime = 0
        self.eval = eval
        self.diminishing_value = diminishing_value

        if self.late_delivery:
            print("Proxy Late Delivery active")
        if self.downtime:
            print("Proxy Downtime active")
        if self.downtime_total:
            print("Proxy Downtime total active")
        if self.wip:
            print("Proxy WIP active")

        self.selected_scenario = scenario

        self.time = 0
        # List of all busy machines
        self.busy_machines = []
        # List of all machines that actually can be allocated a job
        self.legal_machines = []
        # List of all unoccupied machines
        self.machines = []
        # List of all jobs which are not done yet
        self.pending_jobs = []
        # Longest operation duration in all jobs
        self.max_time_operation = 0
        # List of all jobs which are currently in production
        self.active_jobs = []
        # List of all jobs which are done to calculate the final reward
        self.completed_jobs = []
        self.late_jobs = []

        # Machine Mask (Which are legal over all machines)
        self.mask_machines = []
        # Jobs Mask (Which are legal over all Jobs)
        self.mask_jobs = []

        # Load scenario
        self.load_scenario(self.selected_scenario)

        # Maximum payment from the jobs
        self.max_job_value = max([p.job_value for p in self.pending_jobs])
        # All job values combined
        self.max_profit = sum([p.job_value for p in self.pending_jobs])
        # Calculate fix_costs
        self.fix_costs = int((len(self.machines) / 2) * 15)

        # The observation space
        # spaces in the observation space dict
        """
            Dict:
                action_mask: mask of actions which are possible
                real_obs:
                    % of job done: how much of the whole job is done
                    Left over time on current Operation:
                    Time until next machine is not busy: displays the amount of time the job.next_machine for a 
                    specific job will be busy.
                    Legal job: if a job can be legally selected to be worked on 1 True 0 False
                    job value: Monetary value of the job
        """
        spaces = {
            'action_mask': Box(low=0, high=1, shape=(len(self.pending_jobs) + 1,), dtype=np.float64),
            'real_obs': Box(low=0, high=1, shape=(len(self.pending_jobs), 5), dtype=np.float64)
        }

        self.observation_space = Dict(spaces)

        # The action space
        action_space_size = len(self.pending_jobs) + 1
        self.action_space = Discrete(action_space_size)

        # Where does penalties come from
        self.temp_illegal_move = 0
        self.temp_job_ca_penalty = 0
        self.temp_no_legal_action = 0
        self.advance_time_penalty = 0

    def calculate_profit(self):
        """
        Use all the underlying functions to calculate the complete earnings.
        :return:
        """
        profit = 0

        # Calculate the profit from each job
        for complete_job in self.completed_jobs:
            profit += complete_job.calculate_value()

        # Minus fix costs
        profit = profit - (self.time * self.fix_costs)
        return profit

    def calculate_profit_reward(self, profit):
        """
        :return:
        """
        """if profit < 0:
            return (10 * -(-profit)**(1. / 3)).real
        else:
            return (10 * (profit ** (1. / 3))).real"""
        return profit / 10

    def step(self, action):
        """
        Here we need to schedule the jobs to the machines
        :param action:
        :return:
        """
        reward = 0

        # Get all jobs which need to be transferred to the next machine
        awaiting_jobs = [job for job in self.pending_jobs if self.time >= job.step_time > 0]

        # For all jobs which have finished their last machine order
        for job in awaiting_jobs:
            # Get the corresponding machine
            machine = self.machines[job.next_machine]
            # Check if the job is actually done
            machine.check_job_done(self.time, job)
            # Pop the machine from the busy_machine list
            self.busy_machines.pop(self.busy_machines.index(machine))
            # Advance the machine_order of the job
            nb_machining_steps = job.advance_machine_order(self.time)
            # pop job from active_jobs
            self.active_jobs.pop(self.active_jobs.index(job))

            # If there are no more machining steps in machine_order
            if nb_machining_steps == 0:
                self.completed_jobs.append(job)
                # Append late jobs to list for evaluation
                if job.due_time < self.time:
                    if self.late_delivery:
                        reward -= (self.time - job.due_time) * 2
                    self.late_jobs.append(job)
                # Set job complete to True
                job.complete = True

        available_action = self.get_legal_actions()

        # This checks if the agent wants to do nothing
        if action == len(self.pending_jobs):
            sum_machines_not_busy = sum([1 for m in self.machines if not m.busy])
            if self.downtime:
                #if sum_machines_not_busy < len(self.machines)/2:
                #    reward -= sum_machines_not_busy
                #else:
                #    reward += 1
                reward += len(self.machines) - sum_machines_not_busy

            if self.wip:
                num_wip = sum([1 for j in self.pending_jobs if len(j.start_times) > 0])
                if num_wip > len(self.pending_jobs) / 3:
                    reward -= num_wip
                else:
                    reward += 5
            self.num_downtime += sum_machines_not_busy
            reward -= 1
            if sum(self.mask_jobs) > 0:
                #reward -= 1
                self.advance_time_penalty -= 1
            self.advance_time()
        # See if there are any legal actions
        elif available_action:
            # Here the agent has to choose the action
            # Check if the pending jobs machine order fits the machine
            chosen_job = self.pending_jobs[action]
            if not chosen_job.active and not chosen_job.complete:
                # Get corresponding machine
                machine = self.machines[list(chosen_job.machine_order)[0]]
                # Check if machine is busy
                if not machine.busy:
                    machine.start_production(chosen_job)
                    # Set step_time for job
                    chosen_job.set_step_time(self.time)
                    # Set the start time for the job
                    chosen_job.set_start_times(self.time)
                    self.active_jobs.append(chosen_job)

                    # Not needed probably
                    self.busy_machines.append(machine)
                    self.legal_machines.pop(self.legal_machines.index(machine))
                else:
                    # If the agent tries an illegal move his reward will be lowered
                    reward -= 1
                    self.temp_illegal_move -= 1
            else:
                # If the agent chooses a job which is already being worked on or complete
                reward -= 1
                self.temp_job_ca_penalty -= 1
        else:
            # If agent chooses to not advance_time when no legal action is available
            # Alternative just advance_time
            reward -= 1
            self.temp_no_legal_action -= 1

        obs = self._get_observations()
        done = self._is_done()

        info = self._get_info()
        # add the end_reward to reward when last step is done
        if done:
            if self.downtime_total:
                # add total downtime to reward
                reward -= self.num_downtime
            reward += self.calculate_profit_reward(info["profit"])

        return obs, reward, done, info

    def advance_time(self):
        """
        With this function we want to advance the time when no legal action is available or the agent doesnt want to
        take any action. We want this to happen until a legal action is available again.
        :return:
        """
        self.time += 1

    def _is_done(self):
        if len(self.completed_jobs) == len(self.pending_jobs):
            return True
        else:
            return False

    def _get_observations(self):
        self.get_legal_actions()

        # Create real_obs list
        real_obs = [[] for job in self.pending_jobs]

        for index in range(len(self.pending_jobs)):
            selected_job = self.pending_jobs[index]

            # if job is active we need to calculate some observations differently
            if selected_job.active:
                # % of job done
                real_obs[index].append((selected_job.step_time - self.time + sum(list(selected_job.machine_order.values())[1:])) \
                                 / selected_job.total_job_time)
                # Left over time on current operation
                real_obs[index].append((selected_job.step_time - self.time) / self.max_time_operation)
            else:
                # % of job done
                real_obs[index].append(sum(selected_job.machine_order.values()) / selected_job.total_job_time)
                # Left over time on current operation
                real_obs[index].append(0)

            # Time until next machine is not busy
            if selected_job.next_machine is not None:
                machine = self.machines[selected_job.next_machine]
                if machine.working_on is None:
                    # If no job is being worked_on append 0
                    real_obs[index].append(0)
                else:
                    # Calculate remaining time
                    job = machine.working_on
                    real_obs[index].append((job.step_time-self.time) / self.max_time_operation)
            else:
                real_obs[index].append(0)

            # legal job
            real_obs[index].append(self.mask_jobs[index])
            # Monetary Job value
            real_obs[index].append(selected_job.job_value/self.max_job_value)

        # Get the action mask
        action_mask = self.mask_jobs
        # Append a one as the action to skip time is always valid
        if sum(action_mask) > 0:
            action_mask.append(0)
        else:
            action_mask.append(1)

        return_dict = {
            'action_mask': np.array(action_mask),
            'real_obs': np.array(real_obs)
        }
        return return_dict

    def _get_info(self):
        return {
            "profit": self.calculate_profit(),
            "complete_jobs": [j.complete for j in self.pending_jobs]
        }

    def get_legal_actions(self):
        """
        This function should see if there are any legal actions which can be done.
        :return:
        """
        temp_legal_machines = []
        needed_machines = [job.next_machine for job in self.pending_jobs]
        # For every machine in machines
        for machine in self.machines:
            if machine.id in needed_machines and not machine.busy and machine not in temp_legal_machines:
                temp_legal_machines.append(machine)

        # copy temp_legal_machines to global variable
        self.legal_machines = temp_legal_machines

        # Fill mask_machines
        self.mask_machines = [0 if (not m.busy and m in self.legal_machines) else 1 for m in self.machines]

        # Fill mask_jobs
        temp_mask_jobs = []
        # For all jobs
        for j in self.pending_jobs:
            # if j still needs to be worked on and the next_machine for this job is not busy
            if not j.active and not j.complete and not self.machines[j.next_machine].busy:
                # append True to the temp_mask
                temp_mask_jobs.append(1)
            else:
                # append False to the temp_mask
                temp_mask_jobs.append(0)
        # copy temp_mask_jobs to global variable
        self.mask_jobs = temp_mask_jobs

        # If there is more than 0 legal machines return true
        if len(self.legal_machines) > 0:
            return True
        # If there are no legal machines return False
        else:
            return False

    def reset(self, eval=False):
        self.num_downtime = 0

        self.eval = eval
        self.time = 0
        self.fix_costs = 0
        self.busy_machines = []
        self.legal_machines = []
        self.machines = []
        self.max_time_operation = 0
        self.pending_jobs = []
        self.active_jobs = []
        self.completed_jobs = []
        self.late_jobs = []

        # load scenario
        self.load_scenario(self.selected_scenario)

        # Maximum payment from the jobs
        self.max_job_value = max([p.job_value for p in self.pending_jobs])
        # All job values combined
        self.max_profit = sum([p.job_value for p in self.pending_jobs])
        # Calculate fix_costs
        self.fix_costs = int((len(self.machines) / 2) * 15)

        self.mask_machines = []
        self.mask_jobs = []

        # Where does penalties come from
        self.temp_illegal_move = 0
        self.temp_job_ca_penalty = 0
        self.temp_no_legal_action = 0
        self.advance_time_penalty = 0

        obs = self._get_observations()

        return obs

    def load_scenario(self, scenario):
        dir_path = str(Path(__file__).parent.absolute()) + "/scenarios/"
        # If no scenario is set import a predefined basic scenario
        if scenario is None and not self.eval:
            dir_path = dir_path + "40x20x4000/"
            scenario = random.choice(os.listdir(dir_path))

        elif self.eval:
            dir_path = dir_path + "40x20x4000Eval/"
            scenario = "evalscenario"

        print("Selected Scenario: ", scenario)

        # open the file
        scenario_file = open(dir_path + scenario, "r")
        line_str = scenario_file.readline()

        # Create machines
        num_jobs, num_machines = line_str.split()
        for x in range(int(num_machines)):
            self.machines.append(Machine())
            self.machines[-1].id = len(self.machines) - 1

        j = 0
        while line_str:
            line_str = scenario_file.readline()
            # Get single line and split data
            job_data = line_str.split()
            # Check if last line has been reached
            if len(job_data) > 0:
                job_details = {"machine_order": {}}
                # Get the machine_order for the job
                for i in range(0, len(job_data), 2):
                    # Get job_details from the job_data
                    job_details["machine_order"][int(job_data[i])] = int(job_data[i + 1])
                    self.max_time_operation = max(self.max_time_operation, int(job_data[i + 1]))
                self.pending_jobs.append(Job(job_details["machine_order"], self.diminishing_value,
                                             (sum(job_details["machine_order"].values())) + random.randint(20, 40)))
                self.pending_jobs[-1].id = len(self.pending_jobs) - 1
            j += 1

    def render(self, mode="human"):
        df = []
        colors = []
        for job in self.pending_jobs:
            colors.append(tuple([random.random(), random.random(), random.random()]))
            for index in range(len(self.machines)):
                machine_step = job.done_machine_order
                start_time = job.start_times[index]

                dict_op = dict()
                dict_op["Machine"] = "Machine {}".format(list(machine_step)[index])
                dict_op["Task"] = "Job {}".format(job.id)
                dict_op["Start"] = start_time
                dict_op["Finish"] = start_time + machine_step[list(machine_step)[index]]

                df.append(dict_op)

        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(df, colors=colors, index_col="Machine", group_tasks=True, show_colorbar=True)
            fig['layout']['xaxis'].update({'type': None})
            fig.update_yaxes(autorange="reversed")
            fig.show()

        return fig

