
import numpy as np
from torch.utils.data import Sampler


# ------------------------------------------------------------
# 无限采样器,多个进程使用同一采样序列
class Inf_Sampler(Sampler):
    def __init__(self, 
    data_source,                            # torch.Dataset型循环查找结构
    process_id: int,                        # 该采样器作用的进程号
    num_processes: int            =1,       # 总共的进程数目
    need_shuffle: bool            =True,    # 是否将采样索引打乱
    random_seed                   =0        # 随机种子
    ) -> None:
        super().__init__(data_source)

        self.data_source = data_source
        self.data_size = len(data_source)
        assert self.data_size > 0, 'Dataset is empty !'

        self.process_id = process_id # 当前进程编号
        self.num_processes = num_processes  # 训练设置的异步进程数
        assert self.num_processes > 0, 'Numbers of sub-processes for Sampler should be larger than 0 !'

        self.sample_orders = np.arange(self.data_size)
        self.rnd = np.random.RandomState(random_seed)
        self.shuffled = need_shuffle
        if self.shuffled:  # 将原有序列洗牌
            self.rnd.shuffle(self.sample_orders)

    def __iter__(self):
        count = 0 # 计数器
        while True:
            idx = count % self.data_size
            if idx % self.num_processes == self.process_id:
                yield self.sample_orders[idx]  # 取与当前进程对应的索引号，避免异步进程重复取值

            count += 1
            if count % self.data_size == 0:  # 取完一轮之后重新洗牌
                if self.shuffled:
                    self.rnd.shuffle(self.sample_orders)


# # --------------------------------------------------
# # 单进程无限采样器
# class Seq_Sampler(Sampler):

#     def __init__(self, data_source, random_seed=666) -> None:
#         super().__init__(data_source)

#         self.data_source = data_source
#         assert hasattr(self.data_source, 'scope'), 'Dataset has No Attribute \'scope\'.'
#         assert hasattr(self.data_source, 'train_indices'), 'Dataset has No Attribute \'train_indices\'.'
#         assert hasattr(self.data_source, 'test_indices'), 'Dataset has No Attribute \'test_indices\'.'

#         if self.data_source.scope == 'train':
#             self.index_max = len(self.data_source.train_indices)
#         elif self.data_source.scope == 'test':
#             self.index_max = len(self.data_source.test_indices)

#         self.sample_orders = np.arange(self.index_max)
#         self.rnd = np.random.RandomState(random_seed)

#     #
#     def __iter__(self):
#         count = 0
#         while True:
#             yield self.sample_orders[count]
#             count += 1
#             if count >= self.index_max:
#                 self.rnd.shuffle(self.sample_orders)
#                 count = 0

#     #
#     def __len__(self) -> int:
#         return self.index_max



# --------------------------------------------------
# 单进程无限随机采样器
class InfRandomSampler(Sampler):

    def __init__(self, data_source, random_seed=666) -> None:
        super().__init__(data_source)

        self.data_source = data_source
        self.index_max = len(data_source)
        self.sample_orders = np.arange(self.index_max)
        self.rnd = np.random.RandomState(random_seed)

    #
    def __iter__(self):
        count = 0
        while True:
            yield self.sample_orders[count]
            count += 1
            if count >= self.index_max:
                self.rnd.shuffle(self.sample_orders)
                count = 0

    #
    def __len__(self) -> int:
        return self.index_max


# --------------------------------------------------
# 单进程无限顺序采样器
class InfSequentialSampler(Sampler):

    def __init__(self, data_source) -> None:
        super().__init__(data_source)

        self.data_source = data_source
        self.index_max = len(data_source)
        self.sample_orders = np.arange(self.index_max)

    #
    def __iter__(self):
        count = 0
        while True:
            yield self.sample_orders[count]
            count += 1
            if count >= self.index_max:
                count = 0

    #
    def __len__(self) -> int:
        return self.index_max