from multiprocessing import Event, Process, Queue
from queue import Empty
from typing import Any

import numpy as np
from cellpose.transforms import normalize_img, random_rotate_and_resize

from classpose.log import get_logger
from classpose.transforms import create_stardist_augmentation, get_config

dataset_logger = get_logger(__name__)


class ClassposeDataset:
    def __init__(
        self,
        data_array: np.ndarray,
        label_array: np.ndarray,
        diameter_array: np.ndarray,
        n_proc: int = 1,
        batch_size: int = 2,
        augment_pipeline_config: str | None = None,
        diam_mean: float = 100,
        rescale: bool = True,
        scale_range: list[float] | None = None,
        bsize: int = 128,
        normalize_params: dict[str, Any] = {"normalize": True},
        augment: bool = True,
    ):
        self.data_array = data_array
        self.label_array = label_array
        self.diameter_array = diameter_array
        self.n_proc = n_proc
        self.batch_size = batch_size
        self.augment_pipeline_config = augment_pipeline_config
        self.diam_mean = diam_mean
        self.rescale = rescale
        self.scale_range = scale_range
        self.bsize = bsize
        self.normalize_params = normalize_params
        self.augment = augment

        assert self.n_proc >= 0, "n_proc must be greater than or equal to 0"
        assert self.batch_size > 0, "batch_size must be greater than 0"

        if self.n_proc > 0:
            self.start_workers()
        else:
            self.start_lists()

    def reset_queue(self, queue: Queue):
        """
        Resets a queue.

        Args:
            queue (Queue): Queue to reset.
        """
        if self.n_proc > 0:
            try:
                while True:
                    queue.get_nowait()
            except Empty:
                pass
        else:
            self.idxs = []

    def start_lists(self):
        """
        Initialises an index list when n_proc == 0.
        """
        self.idxs = []

    def start_workers(self):
        """
        Starts the workers when n_proc > 0.
        """
        pre_q = Queue()
        q = Queue(maxsize=self.n_proc)
        q_out = Queue()
        stop_event = Event()
        processes = []
        for _ in range(self.n_proc):
            p = Process(
                target=self.work,
                args=(q, q_out),
                name=f"dataset_worker_{_}",
            )
            p.daemon = True
            p.start()
            processes.append(p)

        # pre_q ensures that all items are added at the beginning of
        # each epoch and then processed sequentially by the worker
        # taking elements from q
        self.queue_process = Process(target=self.put_worker, args=(pre_q, q))
        self.queue_process.start()

        self.pre_q = pre_q
        self.q = q
        self.q_out = q_out
        self.stop_event = stop_event
        self.processes = processes

    def augment_image(
        self,
        imgs: np.ndarray,
        lbls: np.ndarray,
        diams: np.ndarray,
        diam_mean: float,
        rescale: bool,
        scale_range: list[float] | None,
        bsize: int,
        normalize_params: dict[str, Any],
        augment: bool,
        augment_pipeline: Any,
    ):
        """
        Augments images and labels.

        Args:
            imgs (np.ndarray): Images to augment.
            lbls (np.ndarray): Labels to augment.
            diams (np.ndarray): Diameters of the images.
            diam_mean (float): Mean diameter of the images.
            rescale (bool): Whether to rescale the images.
            scale_range (list[float] | None): Range of scales to use for rescaling
                augmentation.
            bsize (int): Batch size.
            normalize_params (dict[str, Any]): Parameters for normalization.
            augment (bool): Whether to augment the images.
            augment_pipeline (Any): Augmentation pipeline.

        Returns:
            tuple[np.ndarray, np.ndarray]: Augmented images and labels.
        """
        rsc = diams / diam_mean if rescale else np.ones(diams.shape, "float32")
        # augmentations
        if augment:
            if augment_pipeline is not None:
                imgs = augment_pipeline.transform(imgs)
            imgi, lbl = random_rotate_and_resize(
                [imgs],
                Y=[lbls],
                rescale=[rsc],
                scale_range=scale_range,
                xy=(bsize, bsize),
            )[:2]
        else:
            imgi = imgs
            lbl = lbls
        imgi = normalize_img(imgi, **normalize_params)
        return imgi, lbl

    def work(self, q: Queue, q_out: Queue):
        """
        Worker responsible for processing the queue.

        Args:
            q (Queue): Queue to process.
            q_out (Queue): Queue to output results to.
        """
        augment_pipeline = create_stardist_augmentation(
            get_config(self.augment_pipeline_config)
        )
        while True:
            idx = q.get()
            if idx is None:
                break
            q_out.put(
                self.augment_image(
                    self.data_array[idx],
                    self.label_array[idx][1:],
                    self.diameter_array[idx],
                    diam_mean=self.diam_mean,
                    rescale=self.rescale,
                    scale_range=self.scale_range,
                    bsize=self.bsize,
                    normalize_params=self.normalize_params,
                    augment=self.augment,
                    augment_pipeline=augment_pipeline,
                )
            )

    def put(self, idx: int | list[int], reset_queues: bool = False):
        """
        Adds indices to the queue.

        Args:
            idx (int | list[int]): Index or list of indices to add.
            reset_queues (bool, optional): Whether to reset the pre-queue.
                Defaults to False.
        """
        if reset_queues:
            self.reset_queue(self.pre_q)
        if isinstance(idx, int):
            idx = [idx]
        if self.n_proc > 0:
            # starts a process that sequentially adds items to queue
            # this guarantees that the queue is filled in order
            for i in idx:
                self.pre_q.put(i)
        else:
            self.idxs.extend(idx)

    def put_worker(self, pre_q: Queue, q: Queue):
        """
        Worker responsible for adding to main queue.
        """
        while True:
            idx = pre_q.get()
            if idx is None:
                break
            q.put(idx)

    def get(self):
        """
        Returns a batch from the dataset.
        """
        batch, batch_labels = [], []
        if self.n_proc > 0:
            for _ in range(self.batch_size):
                img, lbl = self.q_out.get()
                batch.append(img)
                batch_labels.append(lbl)
        else:
            for _ in range(self.batch_size):
                idx = self.idxs.pop(0)
                img, lbl = self.augment_image(
                    self.data_array[idx],
                    self.label_array[idx][1:],
                    self.diameter_array[idx],
                    diam_mean=self.diam_mean,
                    rescale=self.rescale,
                    scale_range=self.scale_range,
                    bsize=self.bsize,
                    normalize_params=self.normalize_params,
                    augment=self.augment,
                    augment_pipeline=None,
                )
                batch.append(img)
                batch_labels.append(lbl)
        return np.concatenate(batch), np.concatenate(batch_labels)

    def shutdown(self):
        """
        Shuts down processes if `n_proc` > 0.
        """
        if self.n_proc > 0:
            self.reset_queue(self.pre_q)
            self.reset_queue(self.q)
            self.reset_queue(self.q_out)

            self.pre_q.put(None)

            for p in self.processes:
                self.q.put(None)
                dataset_logger.info(f"Shutting down dataset worker {p.name}")
                p.terminate()
                dataset_logger.info(f"Shut down dataset worker {p.name}")
                p.join()
            dataset_logger.info("Shut down dataset workers")
            self.queue_process.terminate()
            self.queue_process.join()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
