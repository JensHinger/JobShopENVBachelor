import random
from pathlib import Path


# TODO Agent abspeichern potenziell in IPYNB notebook machen
# TODO Agent ausprobieren und evaluations environment schaffen
# TODO Visualization in env erstellen
# TODO Basic scenario erstellen und automatisch lesen von Klasse

def scenario_generator(num_machines, num_jobs, min_time, max_time, total_time, scenario_name):
    # Write first line with num_machines the num_jobs
    assert num_machines > 0
    assert num_jobs > 0
    assert min_time > 0
    assert max_time > min_time
    assert total_time > num_jobs * num_machines
    assert total_time < num_jobs * num_machines * max_time


    file_path = str(Path(__file__).parent.absolute()) + "/scenarios/"
    scenario_file = open(file_path + scenario_name, "w")
    scenario_file.write(f"{num_jobs} {num_machines} \n")
    # Write line for each job? with space as division
    # Create a line for each job
    machine_pool = []
    machine_time = []
    for job in range(num_jobs):
        # Each job has to go through each machine
        machine_pool.append([i for i in range(num_machines)])
        random.shuffle(machine_pool[job])
        machine_time.append([])
        for i in range(num_machines):
            time = min_time + random.randint(min_time, max_time - min_time)

            machine_time[job].append(time)

    sum_time = sum([sum(job) for job in machine_time])
    while sum_time > total_time or sum_time < total_time:
        print(sum_time)

        if sum_time > total_time:
            change_index = random.randint(0, len(machine_time)-1)
            change_index2 = random.randint(0, len(machine_time[change_index])-1)
            job_to_change = machine_time[change_index][change_index2]

            if job_to_change == min_time or job_to_change == max_time:
                pass
            else:
                random_value = random.randint(min_time, job_to_change - min_time)
                while sum_time - random_value < total_time:
                    random_value -= 1
                machine_time[change_index][change_index2] = job_to_change - random_value

        elif sum_time < total_time:
            change_index = random.randint(0, len(machine_time)-1)
            change_index2 = random.randint(0, len(machine_time[change_index])-1)
            job_to_change = machine_time[change_index][change_index2]

            if job_to_change == min_time or job_to_change == max_time:
                pass
            else:
                random_value = random.randint(min_time, max_time - job_to_change)
                while sum_time + random_value > total_time:
                    random_value -= 1

                machine_time[change_index][change_index2] = job_to_change + random_value

        sum_time = sum([sum(job) for job in machine_time])

    print(sum([sum(job) for job in machine_time]))

    # Actually write file
    for job in range(num_jobs):
        for i in range(num_machines):
            scenario_file.write(f"{machine_pool[job][i]} {machine_time[job][i]}   ")

        scenario_file.write("\n")
"""
amount_jobs = 10
amount_machines = 5
min_time = 1
max_time = 10
timesteps = int((int(max_time) / 2) * (amount_machines * amount_jobs))
print("Timestep amount: {}".format(timesteps))
scenario_generator(amount_machines, amount_jobs, min_time, max_time, timesteps, f"{amount_jobs}x{amount_machines}x{timesteps}")
"""
for i in range(5):
    scenario_generator(20, 40, 1, 10, 4000, "40x20x4000/scenario{}".format(i))

scenario_generator(20,40, 1, 10, 4000, "40x20x4000/evalscenario")