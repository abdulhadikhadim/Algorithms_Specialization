import heapq
from collections import namedtuple

AssignedJob = namedtuple("AssignedJob", ["worker", "started_at"])

def assign_jobs(n_workers, jobs):
    result = []
    # Initialize the priority queue (min-heap)
    priority_queue = [(0, i) for i in range(n_workers)]  # (next_free_time, worker_index)
    heapq.heapify(priority_queue)

    for job in jobs:
        # Extract the worker with the earliest next_free_time
        next_free_time, worker_index = heapq.heappop(priority_queue)
        # Assign the job to this worker
        result.append(AssignedJob(worker_index, next_free_time))
        # Update the worker's next_free_time and push it back into the priority queue
        heapq.heappush(priority_queue, (next_free_time + job, worker_index))

    return result

def main():
    n_workers, n_jobs = map(int, input().split())
    jobs = list(map(int, input().split()))
    assert len(jobs) == n_jobs

    assigned_jobs = assign_jobs(n_workers, jobs)

    for job in assigned_jobs:
        print(job.worker, job.started_at)

if __name__ == "__main__":
    main()
