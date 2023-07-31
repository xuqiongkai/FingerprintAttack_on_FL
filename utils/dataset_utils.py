#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# utils for use in the examples and tutorials

import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder, SequentialSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.metrics_reporter import Channel
from flsim.interfaces.model import IFLModel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.data.data_utils import batchify
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

newsgroup_labels = ['18828_rec.sport.baseball', '18828_talk.politics.mideast', '18828_comp.sys.ibm.pc.hardware', '18828_sci.med',
                    '18828_sci.crypt', '18828_talk.politics.guns', '18828_sci.electronics', '18828_soc.religion.christian',
                    '18828_alt.atheism', '18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.mac.hardware', 
                    '18828_comp.windows.x', '18828_misc.forsale', '18828_rec.autos', '18828_rec.motorcycles', 
                    '18828_rec.sport.hockey','18828_sci.space','18828_talk.politics.misc', '18828_talk.religion.misc']

speaker_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 178, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 397, 398, 400, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 416, 417, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599, 600, 603, 604, 605, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 639, 640, 641, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 659, 664, 666, 667, 668, 669, 670, 671, 673, 674, 675, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 727, 728, 730, 731, 732, 734, 736, 737, 738, 739, 740, 741, 742, 743, 746, 748, 750, 751, 752, 753, 754, 755, 756, 757, 759, 760, 761, 762, 763, 764, 765, 766, 767, 769, 771, 772, 773, 774, 775, 777, 778, 779, 780, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 797, 798, 799, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 835, 838, 839, 840, 841, 842, 846, 847, 848, 849, 850, 851, 852, 853]
# size: 43
speaker_idxs_512 = [1, 4, 7, 9, 10, 11, 27, 43, 55, 56, 59, 63, 119, 123, 126, 139, 196, 209, 220, 225, 238, 243, 244, 262, 267, 269, 273, 277, 286, 294, 296, 297, 322, 332, 375, 389, 397, 438, 445, 448, 463, 478, 524]
# size: 65
speaker_idxs_320 = [1, 4, 7, 9, 10, 11, 27, 43, 46, 53, 55, 56, 59, 63, 75, 99, 119, 123, 126, 139, 155, 160, 196, 209, 217, 220, 225, 226, 234, 237, 238, 243, 244, 247, 262, 267, 269, 271, 273, 274, 277, 286, 291, 294, 296, 297, 301, 321, 322, 332, 366, 375, 389, 397, 438, 445, 448, 463, 469, 478, 506, 514, 524, 530, 748]
# size: 70
speaker_idxs_288 = [1, 4, 7, 9, 10, 11, 27, 43, 46, 53, 55, 56, 59, 63, 68, 75, 99, 100, 119, 123, 126, 139, 155, 160, 196, 209, 217, 220, 225, 226, 234, 237, 238, 243, 244, 247, 262, 267, 269, 271, 273, 274, 277, 286, 291, 294, 296, 297, 301, 302, 321, 322, 332, 366, 375, 389, 397, 438, 445, 448, 463, 469, 478, 506, 514, 524, 530, 653, 659, 748]
# size: 72
speaker_idxs_256 = [1, 4, 7, 9, 10, 11, 27, 43, 46, 53, 55, 56, 59, 63, 68, 75, 99, 100, 119, 123, 126, 139, 155, 160, 196, 209, 217, 220, 225, 226, 227, 234, 237, 238, 243, 244, 247, 262, 267, 269, 271, 273, 274, 277, 286, 291, 294, 296, 297, 301, 302, 321, 322, 332, 366, 375, 389, 397, 438, 445, 448, 463, 469, 478, 506, 514, 524, 525, 530, 653, 659, 748]
# size: 99
speaker_idxs_160 = [1, 4, 7, 9, 10, 11, 20, 25, 27, 35, 40, 43, 46, 52, 53, 55, 56, 59, 60, 63, 68, 75, 99, 100, 104, 111, 119, 123, 126, 128, 139, 152, 155, 160, 196, 202, 203, 209, 210, 217, 220, 225, 226, 227, 234, 237, 238, 241, 243, 244, 247, 261, 262, 267, 269, 271, 273, 274, 277, 282, 286, 291, 294, 296, 297, 301, 302, 321, 322, 329, 332, 366, 375, 389, 397, 410, 435, 438, 445, 448, 463, 469, 478, 481, 506, 514, 524, 525, 527, 530, 545, 548, 631, 653, 659, 670, 743, 748, 814]
# size: 119
speaker_idxs_128 = [1, 4, 7, 9, 10, 11, 14, 20, 25, 27, 35, 40, 43, 45, 46, 52, 53, 55, 56, 59, 60, 63, 68, 75, 85, 99, 100, 104, 111, 119, 123, 126, 128, 139, 152, 155, 160, 182, 196, 202, 203, 209, 210, 217, 220, 225, 226, 227, 228, 234, 237, 238, 241, 243, 244, 247, 248, 261, 262, 267, 269, 271, 273, 274, 277, 282, 286, 291, 294, 296, 297, 298, 301, 302, 321, 322, 329, 332, 366, 375, 389, 397, 408, 409, 410, 435, 437, 438, 445, 448, 463, 469, 473, 478, 481, 506, 514, 516, 524, 525, 527, 530, 545, 548, 554, 557, 563, 567, 580, 631, 653, 659, 670, 706, 725, 743, 748, 766, 814]

def collate_fn(batch: Tuple) -> Dict[str, Any]:
    feature, label = batch
    return {"features": feature, "labels": label}


class DataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: VisionDataset,
        eval_dataset: VisionDataset,
        test_dataset: VisionDataset,
        sharder: FLDataSharder,
        batch_size: int,
        drop_last: bool = False,
        collate_fn=collate_fn,
    ):
        assert batch_size > 0, "Batch size should be a positive integer."
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sharder = sharder
        self.collate_fn = collate_fn

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        yield from self._batchify(self.train_dataset, self.drop_last, world_size, rank)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    def _batchify(
        self,
        dataset: VisionDataset,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ) -> Generator[Dict[str, Generator], None, None]:
        # pyre-fixme[16]: `VisionDataset` has no attribute `__iter__`.
        data_rows: List[Dict[str, Any]] = [self.collate_fn(batch) for batch in dataset]
        for _, (_, user_data) in enumerate(self.sharder.shard_rows(data_rows)):
            batch = {}
            keys = user_data[0].keys()
            for key in keys:
                attribute = {
                    key: batchify(
                        [row[key] for row in user_data],
                        self.batch_size,
                        drop_last,
                    )
                }
                batch = {**batch, **attribute}
            yield batch


class UserData(IFLUserData):
    def __init__(self, user_data: Dict[str, Generator], eval_split: float = 0.0):
        self._train_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0

        self._eval_batches = []
        self._num_eval_batches = 0
        self._num_eval_examples = 0

        self._eval_split = eval_split

        user_features = list(user_data["features"])
        user_labels = list(user_data["labels"])
        total = sum(len(batch) for batch in user_labels)

        for features, labels in zip(user_features, user_labels):
            if self._num_eval_examples < int(total * self._eval_split):
                self._num_eval_batches += 1
                self._num_eval_examples += UserData.get_num_examples(labels)
                self._eval_batches.append(UserData.fl_training_batch(features, labels))
            else:
                self._num_train_batches += 1
                self._num_train_examples += UserData.get_num_examples(labels)
                self._train_batches.append(UserData.fl_training_batch(features, labels))

    def num_train_examples(self) -> int:
        """
        Returns the number of train examples
        """
        return self._num_train_examples

    def num_eval_examples(self):
        """
        Returns the number of eval examples
        """
        return self._num_eval_examples

    def num_train_batches(self):
        """
        Returns the number of train batches
        """
        return self._num_train_batches

    def num_eval_batches(self):
        """
        Returns the number of eval batches
        """
        return self._num_eval_batches

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data for training
        """
        for batch in self._train_batches:
            yield batch

    def eval_data(self):
        """
        Iterator to return a user batch data for evaluation
        """
        for batch in self._eval_batches:
            yield batch

    @staticmethod
    def get_num_examples(batch: List) -> int:
        return len(batch)

    @staticmethod
    def fl_training_batch(
        features: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # list to tensors for processing labels/attentions
        return {"features": torch.stack(features), "labels": torch.stack(labels)}


class LEAFDataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.train_dataset, self.drop_last)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    def _batchify(
        self, dataset: Dataset, drop_last=False
    ) -> Generator[Dict[str, Generator], None, None]:
        # pyre-fixme[16]: `Dataset` has no attribute `__iter__`.
        for one_user_inputs, one_user_labels in dataset:
            data = list(zip(one_user_inputs, one_user_labels))
            random.shuffle(data)
            one_user_inputs, one_user_labels = zip(*data)
            batch = {
                "features": batchify(one_user_inputs, self.batch_size, drop_last),
                "labels": batchify(one_user_labels, self.batch_size, drop_last),
            }
            yield batch


class DataProvider(IFLDataProvider):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._train_users = self._create_fl_users(
            data_loader.fl_train_set(), eval_split=0.0
        )
        self._eval_users = self._create_fl_users(
            data_loader.fl_eval_set(), eval_split=1.0
        )
        self._test_users = self._create_fl_users(
            data_loader.fl_test_set(), eval_split=1.0
        )

    def train_user_ids(self) -> List[int]:
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
            )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self._train_users.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self._eval_users.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        for user_data in self._test_users.values():
            yield user_data

    def _create_fl_users(
        self, iterator: Iterator, eval_split: float = 0.0
    ) -> Dict[int, IFLUserData]:
        return {
            user_index: UserData(user_data, eval_split=eval_split)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }


def build_data_provider(
    local_batch_size, examples_per_user, image_size
) -> DataProvider:

    # 1. Create training, eval, and test datasets like in non-federated learning.
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root="./cifar10", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root="./cifar10", train=False, download=True, transform=transform
    )

    # 2. Create a sharder, which maps samples in the training data to clients.
    sharder = SequentialSharder(examples_per_shard=examples_per_user)

    # 3. Shard and batchify training, eval, and test data.
    fl_data_loader = DataLoader(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        test_dataset=test_dataset,
        sharder=sharder,
        batch_size=local_batch_size,
        drop_last=False,
    )

    # 4. Wrap the data loader with a data provider.
    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider

def cross_entropy_eval(lm_logits, labels):
    """
    Routine from Huggingface's GPT-2 implementation (v 4.7.0)
    """
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    XH = torch.nn.CrossEntropyLoss()
    return XH(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class FLModel(IFLModel):
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device

    def fl_forward(self, batch) -> FLBatchMetrics:
        features = batch["features"]
        labels = batch["labels"]
        # transform to Huggingface batch format
        if self.device is not None:
            features = features.to(self.device)
            labels = labels.to(self.device)
        
        # code for transferred GPT-2
        logits = self.model(features)
        loss = cross_entropy_eval(logits, labels)
        
        # code for original HuggingFace GPT-2
        # batch = {'input_ids': features, 'attention_mask': labels, 'labels': features}
        # outputs = self.model(**batch)
        # loss = outputs.loss
        
        num_examples = self.get_num_examples(batch)
        # logits = outputs.logits.detach().cpu()
        # attentions = labels.detach().cpu()
        del features
        del labels
        del logits
        
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=loss,
            targets=None,
            model_inputs=None
        )
        
    def fl_create_training_batch(self, **kwargs):
        features = kwargs.get("features", None)
        labels = kwargs.get("labels", None)
        return UserData.fl_training_batch(features, labels)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)  # pyre-ignore

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return UserData.get_num_examples(batch["labels"])

class LMMetricsReporter(FLMetricsReporter):
    ACCURACY = "Accuracy" # the accuracy is actually loss here

    def __init__(
        self,
        channels: List[Channel],
        target_eval: float = float('inf'),
        window_size: int = 5,
        average_type: str = "sma",
        log_dir: Optional[str] = None,
    ):
        super().__init__(channels, log_dir)
        self.set_summary_writer(log_dir=log_dir)
        self._round_to_target = float(1e10)

    def compare_metrics(self, eval_metrics, best_metrics):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True

        current_accuracy = eval_metrics.get(self.ACCURACY, float("-inf"))
        best_accuracy = best_metrics.get(self.ACCURACY, float("-inf"))
        return current_accuracy < best_accuracy

    def compute_scores(self) -> Dict[str, Any]:
        # compute avg loss
        avg_loss = sum(self.predictions_list).item() / len(self.predictions_list)
        return {self.ACCURACY: avg_loss}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        accuracy = scores[self.ACCURACY]
        return {self.ACCURACY: accuracy}
