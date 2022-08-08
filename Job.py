class Job:


    def __init__(self, machine_order, diminishing_late, job_time):
        # Add a id to each job
        self.id = 0
        self.active = False
        self.complete = False
        self.machine_order = machine_order
        self.next_machine = list(machine_order)[0]
        self.next_machine_time = self.machine_order[self.next_machine]
        self.diminishing_late = diminishing_late
        self.job_time = job_time
        self.due_time = 0

        self.total_job_time = sum(list(self.machine_order.values()))

        # Calculate Job_value
        self.job_value = self.total_job_time * 20
        self.step_time = 0
        self.completion_time = None

        # Variables to render
        self.start_times = []
        self.done_machine_order = {}

    def calculate_value(self):
        """
         Currently we calculate that a contract loses a
         sepcific percentage of its value each time step after the due date
        :return:
        """
        if self.completion_time > self.due_time:
            return self.job_value * self.diminishing_late ** self.completion_time - self.due_time
        else:
            return self.job_value

    def set_completion_time(self, completion_time):
        """
        Handles everything that happens when the production is finalised
        :param completion_time:
        :return:
        """
        self.complete = True
        self.completion_time = completion_time

    def set_start_times(self, start_time):
        self.start_times.append(start_time)

    def set_step_time(self, time):
        """
        Sets time when the step will be finished
        :param step_time:
        :return:
        """
        if self.due_time == 0:
            self.due_time = int(self.total_job_time + 0.25 * self.total_job_time)
        self.step_time = time + self.next_machine_time

    def advance_machine_order(self, time):
        """
        Handles the advancement of the machining order
        :return:
        """
        # pop old machine_order
        if len(list(self.machine_order)) > 0:
            self.done_machine_order[list(self.machine_order)[0]] = self.machine_order[list(self.machine_order)[0]]
            self.machine_order.pop(list(self.machine_order)[0])
        # If there is atleast one more machine_order
        if len(list(self.machine_order)) > 0:
            # set a new next_machine/time to that next order
            self.next_machine = list(self.machine_order)[0]
            self.next_machine_time = self.machine_order[self.next_machine]
        else:
            # When job doesnt have a next machine to go through
            self.set_completion_time(time)
            # calculate value of the contract
            self.calculate_value()
            # Set next machine/time to nona as the product is done
            self.next_machine_time = None
            self.next_machine = None

        # Return amount of orders still ahead
        return len(list(self.machine_order))
