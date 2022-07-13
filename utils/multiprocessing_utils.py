import copy
import multiprocessing as mp
import time
from threading import Thread
from collections import OrderedDict
import abc
from typing import List, Optional, Union, Dict, Any, Tuple
from setproctitle import setproctitle as ptitle

from allenact.utils.system import get_logger, init_logging


class Worker(abc.ABC):
    wid: int
    gpu: Optional[int]
    env: Optional[Any]
    _die_on_exception: bool = False

    def __init__(
        self,
        wid: int,
        gpu: Optional[int],
        job_queue: mp.Queue,
        result_queue: mp.Queue,
        env_args: Dict[str, Any],
        die_on_exception: bool = False,
        verbose: bool = True,
    ):
        self.wid = wid
        self.gpu = gpu
        self.env = None
        self._die_on_exception = die_on_exception

        ptitle(self.info_header)

        ntasks = 0

        try:
            self.env = self.create_env(**env_args)

            while True:
                job = job_queue.get()
                if job is None:
                    break

                if verbose:
                    get_logger().info(
                        f"{self.info_header}: Received {self.job_str(job)}"
                    )

                success = False
                error_count = 0
                while not success:
                    try:
                        res = self.work(job[0], job[2])
                        result_queue.put((job[0], job[1], True, job[2], res))
                        success = True
                    except Exception as e:
                        error_count += 1

                        get_logger().exception(
                            f"{self.info_header}: Exception when processing {self.job_str(job)}"
                        )

                        if self._die_on_exception:
                            result_queue.put((job[0], job[1], False, job[2], None))
                            raise e

                        no_env = True
                        no_env_count = 0
                        while no_env:
                            try:
                                self.destroy_env()
                                self.env = self.create_env(**env_args)
                                no_env = False
                            except Exception as e:
                                no_env_count += 1
                                if no_env_count == 5:
                                    # We're going to stop this worker
                                    result_queue.put(
                                        (job[0], job[1], False, job[2], None)
                                    )
                                    raise e
                                get_logger().exception(f"{self.info_header}:")

                        if error_count == 5:
                            get_logger().exception(
                                f"{self.info_header}: Skipping task after {error_count} failures"
                            )
                            result_queue.put((job[0], job[1], False, job[2], None))
                            break  # success is False, but we stop trying

                if verbose:
                    get_logger().info(
                        f"{self.info_header}: Dispatched {self.job_str(job, max_len=3)}"
                    )

                ntasks += 1
        finally:
            try:
                self.destroy_env()
            except:
                get_logger().exception(
                    f"{self.info_header}: Exception when closing env"
                )

            get_logger().info(f"{self.info_header}: DONE, processed {ntasks} jobs")

    @property
    def info_header(self) -> str:
        if self.gpu is not None:
            return f"{type(self).__name__} {self.wid} GPU {self.gpu}"
        else:
            return f"{type(self).__name__} {self.wid}"

    def job_str(
        self,
        job: Union[Tuple[Optional[str], int, Dict[str, Any]], Dict[str, Any]],
        max_len: int = 300,
    ):
        max_len = max(max_len, 3)

        if isinstance(job, Dict):
            job = copy.deepcopy(job)
            job = (job.pop("task_type", "UNK_TASK"), job.pop("task_type_id", "-1"), job)

        info_str = f"{job[2]}"
        if len(info_str) > max_len:
            info_str = info_str[: max_len - 3] + "..."

        return f"{job[0]} {job[1]} {info_str}"

    def destroy_env(self):
        if self.env is not None and callable(getattr(self.env, "stop", None)):
            self.env.stop()

    @abc.abstractmethod
    def create_env(self, **env_args: Any) -> Optional[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def work(
        self, task_type: Optional[str], task_info: Dict[str, Any]
    ) -> Optional[Any]:
        raise NotImplementedError


class Manager(abc.ABC):
    _write_queue: mp.Queue
    _read_queue: mp.Queue

    _emitted: Dict[Optional[str], int] = OrderedDict()
    _pending_info: Dict[Tuple[Optional[str], int], Dict[str, Any]] = {}
    _received: Dict[Optional[str], int] = OrderedDict()

    env_args: Dict[str, Any]
    num_gpus: int

    def __init__(
        self,
        worker_class: type,
        env_args: Optional[Dict[str, Any]] = None,
        workers: int = 10,
        ngpus: int = 2,
        die_on_exception: bool = False,
        verbose: bool = True,
        debugging: bool = False,
        log_level: str = "info",
        sleep_between_workers: float = 0.0,
    ):
        self.debugging = debugging
        init_logging("debug" if self.debugging else log_level)

        ptitle(self.info_header)

        process_class = mp.Process if not debugging else Thread

        self.env_args = env_args or {}
        self.verbose = verbose or debugging
        self.workers = workers if not debugging else 1
        self.num_gpus = ngpus if not debugging else 0
        self._first_tasks_needed = True

        processes = []

        if self.workers == 0:
            get_logger().info(f"{self.info_header}: DONE")
            return

        self._write_queue = mp.Queue()
        self._read_queue = mp.Queue()

        # We could also just start processing after all workers are active.
        # Just a little bit slower at start but we get rid of the additional thread.
        _processor = Thread(target=self._run_process, daemon=True)
        _processor.start()
        get_logger().info(f"{self.info_header}: processing started")

        get_logger().info(f"{self.info_header}: Starting {self.workers} workers")
        for i in range(self.workers):
            worker_args = dict(
                wid=i,
                gpu=i % self.num_gpus if self.num_gpus > 0 else None,
                job_queue=self._write_queue,
                result_queue=self._read_queue,
                env_args=self.env_args,
                die_on_exception=die_on_exception or debugging,
                verbose=verbose,
            )
            p = process_class(target=worker_class, kwargs=worker_args)
            p.start()
            processes.append(p)
            if i % 10 == 0 and i < self.workers - 1:
                get_logger().info(
                    f"{self.info_header}: {len(processes)} workers started"
                )
            if sleep_between_workers > 0 and i < self.workers - 1:
                time.sleep(sleep_between_workers)

        get_logger().info(f"{self.info_header}: {len(processes)} workers started")

        _processor.join()
        get_logger().info(f"{self.info_header}: processing terminated")

        for _ in range(len(processes)):
            self._write_queue.put(None)  # 1 kill pill for each worker

        for p in processes:
            p.join()
        get_logger().info(f"{self.info_header}: {len(processes)} workers terminated")

        get_logger().info(f"{self.info_header}: DONE")

    def log_status(self):
        message_str = f"{self.info_header}:"
        for entry in self._emitted:
            message_str += (
                f" {entry} out {self._emitted[entry]} in {self._received[entry]},"
            )
        get_logger().info(message_str)

    @property
    def first_tasks_needed(self):
        return self._first_tasks_needed

    @property
    def num_emitted(self):
        return sum([v for k, v in self._emitted.items()])

    @property
    def num_received(self):
        return sum([v for k, v in self._received.items()])

    @property
    def num_pending(self):
        return len(self._pending_info)

    @property
    def info_header(self) -> str:
        return f"{type(self).__name__}"

    def enqueue(
        self, task_info: Dict[str, Any], task_type: Optional[str] = None
    ) -> None:
        if task_type not in self._emitted:
            self._emitted[task_type] = 1
            self._received[task_type] = 0
        else:
            self._emitted[task_type] += 1

        task_type_id = self._emitted[task_type] - 1

        self._write_queue.put((task_type, task_type_id, task_info))

        assert (
            task_type,
            task_type_id,
        ) not in self._pending_info, f"BUG: duplicate task {(task_type, task_type_id)}"
        self._pending_info[(task_type, task_type_id)] = dict(
            task_info=task_info, stime=time.time()
        )

    def dequeue(self) -> Tuple[Optional[str], bool, Dict[str, Any], Any]:
        task_type, task_type_id, success, task_info, result = self._read_queue.get()
        self._received[task_type] += 1
        self._pending_info.pop((task_type, task_type_id))
        return task_type, success, task_info, result

    @property
    def all_work_done(self):
        return len(self._emitted) > 0 and len(self._pending_info) == 0

    @property
    def pending_tasks(self) -> List[Tuple[Optional[str], int, Dict[str, Any]]]:
        order = list(self._emitted.keys())
        return sorted(
            [
                (task_type, task_type_id, task_info)
                for (task_type, task_type_id), task_info in self._pending_info.items()
            ],
            key=lambda x: (order.index(x[0]), x[1]),
        )

    def _run_process(self, failures_to_discard: int = 5) -> None:
        self.work(None, {}, True, None)
        self._first_tasks_needed = False

        while not self.all_work_done:
            if self.verbose:
                self.log_status()
            task_type, success, task_info, result = self.dequeue()  # from _read_queue
            work_failures = 1
            while 0 < work_failures <= failures_to_discard:
                try:
                    self.work(task_type, task_info, success, result)
                    work_failures = 0
                except:
                    get_logger().exception(
                        f"{self.info_header}: {work_failures}-th failure"
                    )
                    work_failures += 1
            if work_failures > failures_to_discard:
                get_logger().error(
                    f"{self.info_header}: Failed to process task"
                    f" of type {task_type}"
                    f" with success {success}"
                    f" task_info {task_info}"
                    f" and result {result}"
                )

        self.log_status()

    @abc.abstractmethod
    def work(
        self,
        task_type: Optional[str],
        task_info: Dict[str, Any],
        success: bool,
        result: Any,
    ) -> None:
        """
        Use `self.enqueue(next_task_info[, next_task_type])` for each new task to be enqueued.
        If `self.first_tasks_needed` is True, `work(...)` is expected to enqueue new tasks.
        If `self.all_work_done` is True, nothing is left to be processed beyond the current result.
        """
        raise NotImplementedError
