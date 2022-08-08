class Machine:

    def __init__(self):
        self.id = 0
        self.busy = False
        self.working_on = None

    def start_production(self, job):
        """
        Handles the start of the production
        :param job:
        :return:
        """
        self.working_on = job
        job.active = True
        self.busy = True

    def check_job_done(self, time, job):
        """
        Checks if the current active job is done
        Calls the finish_production function when job is done
        :param time:
        :return:
        """
        if self.working_on == job and time == self.working_on.completion_time or time == self.working_on.step_time:
            self.finish_production()
            return True
        else:
            return False

    def finish_production(self):
        """
        Handles everything that happens when a production has finished
        :return:
        """
        self.working_on.active = False
        self.working_on.step_time = 0
        self.working_on = None
        self.busy = False
