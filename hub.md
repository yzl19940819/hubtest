## 打印Paddle版本和PaddleHub版本


```python
import paddle
import paddlehub as hub
print(paddle.__version__)
print(hub.__version__)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


    2.0.1
    2.0.4


## 解压数据集，并打印前三后三行


```python
! tar -xzvf data/data16287/thu_news.tar.gz
```

    thu_news/
    thu_news/test.txt
    thu_news/valid.txt
    thu_news/train.txt



```python
!head -n 3 thu_news/valid.txt
```

    text_a	label
    直击特里勾手助小牛反超神鸟发威火箭仍处劣势　　新浪体育讯　北京时间4月16日消息，火箭今天迎来常规赛的收官战。客场挑战达拉斯小牛的比赛将关系到火箭队最终的排名，目前西部的竞争仍然非常激励。尤其是西部第二到第七的六支球队将在今天展开捉对厮杀的情况下，黄蜂、小牛、开拓者以及马刺都有可能是火箭队的下一个对手。以下为本场比赛的精彩瞬间——　　第四节比赛还剩下8分多钟，姚明重新回到场上，但是小牛队的霍里随后在右翼接队友传球后上演一记三分跳投，虽然身体动作已经变形，但是他仍然将球命中，随后基德在弧顶处的一记三分跳投帮助小牛终于将比分反超，火箭在进攻中直接打不开局面，而小牛更是利用特里空切后的一记勾手将比分反超至4分，包括库班在内的所有小牛现场球迷都站起来振臂高呼，而此时阿德尔曼还在场边低头深沉的思索中。　　比赛还剩下最后5分多钟，洛瑞带球突破中被小牛队员犯规，洛瑞来不及刹车让自己直接撞到了观众席上，右腿疼痛难忍的他脸都几乎变形，结果他两罚两中，火箭队还落后两分。随后，洛瑞将球直接传给此时已经回到场上的阿泰，野兽此时将球稳下，并没有急于进攻，随后他顺势将球塞给兰德里，神鸟利用对方站位空隙直接杀到篮下，上篮成功，火箭将比分终于追至80平。场上局面对于火箭来说绝对是不容有失。　　(sabrina)	体育
    北京网购年消费额112亿元　　商报讯(记者吴文治)昨日，淘宝网发布的《2009-2010年度中国网购热门城市报告》显示，北京年度网购消费力达112.5亿元，与上海相差近62亿元，位列十大热门消费力城市第二位。此外，男性网购的消费金额高出女性，与“女性是网购主力军”的传统观念不符。　　淘宝公布的报告显示，中国网购消费力十大城市分别是上海、北京、深圳、杭州、广州、南京、苏州、天津、温州和宁波。主要集中在以江浙沪为主的长三角地区、以广深为主的珠三角地区和以北京为主的京津地区。北京年度网购112.5亿元的消费额，占国内城市网购消费额的5.6%。　　中国网购消费力十大城市的消费金额性别来源比例中，男性占比超过了女性。前者占比达到53.5%，后者则为46.5%。不过，在成交人数、成交笔数等关键数据上显示，女性消费者均高于男性。此外，在十大网购热门城市中，25岁-34岁的群体成为网购消费的主力军，占比达到62.49%。	科技



```python
!tail -n 3 thu_news/valid.txt
```

    事业测试：你工作易受他人干扰吗(图)　　独家撰稿：苑椿　心理测试征稿启事 欢迎关注新浪星座微博　　办公室永远是个龙蛇混杂、藏龙卧虎的地方，你永远不知道一张张面具底下会是怎样的脸庞，你是否还傻傻的对别人的话听之任之，完全搞乱了自己工作的步调？还是笃定的坚守阵地，从未被谣言动摇分毫？赶紧来测测看吧！　　(本测试仅供娱乐，非专业心理指导。)	星座
    趣味测试：你怎么红红火火过春节(图)　　独家撰稿：岚　心理测试征稿启事 欢迎关注新浪星座微博　　红红火火过大年啦，每年的此时你都会如何度过呢？是跟家人爱人在一起还是跟朋友兄弟外出欢聚呢？亦或背起行囊远离嘈杂，无论如何，总会有一种适合你的方式，赶紧来测测看吧！　　(本测试仅供娱乐，非专业心理指导。)	星座
    人际测试：你的人际磁场强大吗(图)　　独家撰稿：大智若笨　心理测试征稿启事　　如何在草木皆兵的office里脱颖而出，最好的办法不是到处抱怨也不是埋头工作，而是要加强自己的磁场，让周围的人都被你所感染，如此有影响力，你难道还怕自己不能夺人眼球吗。　　独家撰稿：大智若笨　心理测试征稿启事　　如何在草木皆兵的office里脱颖而出，最好的办法不是到处抱怨也不是埋头工作，而是要加强自己的磁场，让周围的人都被你所感染，如此有影响力，你难道还怕自己不能夺人眼球吗。	星座


## 数据集准备


```python
import os, io, csv
from paddlehub.datasets.base_nlp_dataset import InputExample, TextClassificationDataset

class ThuNews(TextClassificationDataset):
    def __init__(self, tokenizer, mode='train', max_seq_len=128):
        if mode == 'train':
            data_file = 'train.txt'
        elif mode == 'test':
            data_file = 'test.txt'
        else:
            data_file = 'valid.txt'
        super(ThuNews, self).__init__(
            base_path='thu_news',
            data_file=data_file,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            is_file_with_header=True,
            label_list=['体育', '科技', '社会', '娱乐', '股票', '房产', '教育', '时政', '财经', '星座', '游戏', '家居', '彩票', '时尚'])

    # 解析文本文件里的样本
    def _read_file(self, input_file, is_file_with_header: bool = False):
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                examples = []
                seq_id = 0
                header = next(reader) if is_file_with_header else None
                for line in reader:
                    example = InputExample(guid=seq_id, text_a=line[0], label=line[1])
                    seq_id += 1
                    examples.append(example)
                return examples
```

## 加载模型并实例化数据集


```python
# 模型加载
model = hub.Module(name='ernie', task='seq-cls', num_classes=14)
tokenizer = model.get_tokenizer()
```

    [2021-04-14 00:58:41,175] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-1.0/ernie_v1_chn_base.pdparams
    [2021-04-14 00:58:42,706] [    INFO] - Found /home/aistudio/.paddlenlp/models/ernie-1.0/vocab.txt



```python
# 实例化数据集
train_dataset = ThuNews(tokenizer, mode='train')
dev_dataset = ThuNews(tokenizer, mode='dev')

```

## 模型训练


```python
# 模型训练
optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='ckpt', use_gpu=True)
trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset, save_interval=1)
```

    [2021-04-14 01:01:02,054] [ WARNING] - PaddleHub model checkpoint not found, start from scratch...
    [2021-04-14 01:01:04,458] [   TRAIN] - Epoch=1/3, Step=10/282 loss=2.2694 acc=0.3125 lr=0.000050 step/sec=4.20 | ETA 00:03:21
    [2021-04-14 01:01:06,628] [   TRAIN] - Epoch=1/3, Step=20/282 loss=1.6079 acc=0.5625 lr=0.000050 step/sec=4.61 | ETA 00:03:12
    [2021-04-14 01:01:08,792] [   TRAIN] - Epoch=1/3, Step=30/282 loss=1.0819 acc=0.7438 lr=0.000050 step/sec=4.62 | ETA 00:03:09
    [2021-04-14 01:01:10,964] [   TRAIN] - Epoch=1/3, Step=40/282 loss=0.7970 acc=0.8187 lr=0.000050 step/sec=4.60 | ETA 00:03:07
    [2021-04-14 01:01:13,142] [   TRAIN] - Epoch=1/3, Step=50/282 loss=0.6111 acc=0.8500 lr=0.000050 step/sec=4.59 | ETA 00:03:07
    [2021-04-14 01:01:15,322] [   TRAIN] - Epoch=1/3, Step=60/282 loss=0.6754 acc=0.8063 lr=0.000050 step/sec=4.59 | ETA 00:03:06
    [2021-04-14 01:01:17,502] [   TRAIN] - Epoch=1/3, Step=70/282 loss=0.5108 acc=0.8719 lr=0.000050 step/sec=4.59 | ETA 00:03:06
    [2021-04-14 01:01:19,678] [   TRAIN] - Epoch=1/3, Step=80/282 loss=0.4940 acc=0.8750 lr=0.000050 step/sec=4.60 | ETA 00:03:06
    [2021-04-14 01:01:21,852] [   TRAIN] - Epoch=1/3, Step=90/282 loss=0.3264 acc=0.9375 lr=0.000050 step/sec=4.60 | ETA 00:03:05
    [2021-04-14 01:01:24,029] [   TRAIN] - Epoch=1/3, Step=100/282 loss=0.3493 acc=0.9281 lr=0.000050 step/sec=4.59 | ETA 00:03:05
    [2021-04-14 01:01:26,208] [   TRAIN] - Epoch=1/3, Step=110/282 loss=0.2941 acc=0.9406 lr=0.000050 step/sec=4.59 | ETA 00:03:05
    [2021-04-14 01:01:28,394] [   TRAIN] - Epoch=1/3, Step=120/282 loss=0.3926 acc=0.8969 lr=0.000050 step/sec=4.57 | ETA 00:03:05
    [2021-04-14 01:01:30,580] [   TRAIN] - Epoch=1/3, Step=130/282 loss=0.3005 acc=0.9156 lr=0.000050 step/sec=4.57 | ETA 00:03:05
    [2021-04-14 01:01:32,772] [   TRAIN] - Epoch=1/3, Step=140/282 loss=0.1950 acc=0.9469 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:01:34,958] [   TRAIN] - Epoch=1/3, Step=150/282 loss=0.3482 acc=0.9062 lr=0.000050 step/sec=4.58 | ETA 00:03:05
    [2021-04-14 01:01:37,145] [   TRAIN] - Epoch=1/3, Step=160/282 loss=0.2942 acc=0.9250 lr=0.000050 step/sec=4.57 | ETA 00:03:05
    [2021-04-14 01:01:39,340] [   TRAIN] - Epoch=1/3, Step=170/282 loss=0.3448 acc=0.9000 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:01:41,532] [   TRAIN] - Epoch=1/3, Step=180/282 loss=0.2653 acc=0.9344 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:01:43,724] [   TRAIN] - Epoch=1/3, Step=190/282 loss=0.3599 acc=0.9031 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:01:45,925] [   TRAIN] - Epoch=1/3, Step=200/282 loss=0.2475 acc=0.9281 lr=0.000050 step/sec=4.54 | ETA 00:03:05
    [2021-04-14 01:01:48,116] [   TRAIN] - Epoch=1/3, Step=210/282 loss=0.3296 acc=0.9031 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:01:50,308] [   TRAIN] - Epoch=1/3, Step=220/282 loss=0.2207 acc=0.9250 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:01:52,497] [   TRAIN] - Epoch=1/3, Step=230/282 loss=0.3067 acc=0.9062 lr=0.000050 step/sec=4.57 | ETA 00:03:05
    [2021-04-14 01:01:54,693] [   TRAIN] - Epoch=1/3, Step=240/282 loss=0.2825 acc=0.9250 lr=0.000050 step/sec=4.55 | ETA 00:03:05
    [2021-04-14 01:01:56,887] [   TRAIN] - Epoch=1/3, Step=250/282 loss=0.3720 acc=0.8812 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:01:59,079] [   TRAIN] - Epoch=1/3, Step=260/282 loss=0.2520 acc=0.9281 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:02:01,273] [   TRAIN] - Epoch=1/3, Step=270/282 loss=0.2673 acc=0.9187 lr=0.000050 step/sec=4.56 | ETA 00:03:05
    [2021-04-14 01:02:03,461] [   TRAIN] - Epoch=1/3, Step=280/282 loss=0.2803 acc=0.9281 lr=0.000050 step/sec=4.57 | ETA 00:03:05
    [2021-04-14 01:02:07,233] [    EVAL] - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - [Evaluation result] avg_acc=0.9050
    [2021-04-14 01:02:20,215] [    EVAL] - Saving best model to ckpt/best_model [best acc=0.9050]
    [2021-04-14 01:02:20,218] [    INFO] - Saving model checkpoint to ckpt/epoch_1
    [2021-04-14 01:02:35,298] [   TRAIN] - Epoch=2/3, Step=10/282 loss=0.1729 acc=0.9594 lr=0.000050 step/sec=0.38 | ETA 00:04:30
    [2021-04-14 01:02:37,477] [   TRAIN] - Epoch=2/3, Step=20/282 loss=0.1107 acc=0.9812 lr=0.000050 step/sec=4.59 | ETA 00:04:27
    [2021-04-14 01:02:39,641] [   TRAIN] - Epoch=2/3, Step=30/282 loss=0.1532 acc=0.9594 lr=0.000050 step/sec=4.62 | ETA 00:04:24
    [2021-04-14 01:02:41,809] [   TRAIN] - Epoch=2/3, Step=40/282 loss=0.1967 acc=0.9500 lr=0.000050 step/sec=4.61 | ETA 00:04:22
    [2021-04-14 01:02:43,981] [   TRAIN] - Epoch=2/3, Step=50/282 loss=0.1314 acc=0.9656 lr=0.000050 step/sec=4.60 | ETA 00:04:19
    [2021-04-14 01:02:46,159] [   TRAIN] - Epoch=2/3, Step=60/282 loss=0.1965 acc=0.9469 lr=0.000050 step/sec=4.59 | ETA 00:04:17
    [2021-04-14 01:02:48,338] [   TRAIN] - Epoch=2/3, Step=70/282 loss=0.1583 acc=0.9563 lr=0.000050 step/sec=4.59 | ETA 00:04:15
    [2021-04-14 01:02:50,518] [   TRAIN] - Epoch=2/3, Step=80/282 loss=0.1751 acc=0.9531 lr=0.000050 step/sec=4.59 | ETA 00:04:13
    [2021-04-14 01:02:52,699] [   TRAIN] - Epoch=2/3, Step=90/282 loss=0.1675 acc=0.9563 lr=0.000050 step/sec=4.58 | ETA 00:04:11
    [2021-04-14 01:02:54,862] [   TRAIN] - Epoch=2/3, Step=100/282 loss=0.1208 acc=0.9688 lr=0.000050 step/sec=4.62 | ETA 00:04:09
    [2021-04-14 01:02:57,041] [   TRAIN] - Epoch=2/3, Step=110/282 loss=0.2012 acc=0.9375 lr=0.000050 step/sec=4.59 | ETA 00:04:08
    [2021-04-14 01:02:59,220] [   TRAIN] - Epoch=2/3, Step=120/282 loss=0.1598 acc=0.9594 lr=0.000050 step/sec=4.59 | ETA 00:04:06
    [2021-04-14 01:03:01,393] [   TRAIN] - Epoch=2/3, Step=130/282 loss=0.1068 acc=0.9625 lr=0.000050 step/sec=4.60 | ETA 00:04:04
    [2021-04-14 01:03:03,567] [   TRAIN] - Epoch=2/3, Step=140/282 loss=0.1589 acc=0.9500 lr=0.000050 step/sec=4.60 | ETA 00:04:03
    [2021-04-14 01:03:05,742] [   TRAIN] - Epoch=2/3, Step=150/282 loss=0.1452 acc=0.9656 lr=0.000050 step/sec=4.60 | ETA 00:04:02
    [2021-04-14 01:03:07,927] [   TRAIN] - Epoch=2/3, Step=160/282 loss=0.1363 acc=0.9656 lr=0.000050 step/sec=4.58 | ETA 00:04:00
    [2021-04-14 01:03:10,103] [   TRAIN] - Epoch=2/3, Step=170/282 loss=0.1348 acc=0.9594 lr=0.000050 step/sec=4.60 | ETA 00:03:59
    [2021-04-14 01:03:12,276] [   TRAIN] - Epoch=2/3, Step=180/282 loss=0.1530 acc=0.9563 lr=0.000050 step/sec=4.60 | ETA 00:03:58
    [2021-04-14 01:03:14,448] [   TRAIN] - Epoch=2/3, Step=190/282 loss=0.1605 acc=0.9688 lr=0.000050 step/sec=4.60 | ETA 00:03:57
    [2021-04-14 01:03:16,625] [   TRAIN] - Epoch=2/3, Step=200/282 loss=0.1408 acc=0.9688 lr=0.000050 step/sec=4.59 | ETA 00:03:56
    [2021-04-14 01:03:18,811] [   TRAIN] - Epoch=2/3, Step=210/282 loss=0.1390 acc=0.9594 lr=0.000050 step/sec=4.58 | ETA 00:03:55
    [2021-04-14 01:03:21,010] [   TRAIN] - Epoch=2/3, Step=220/282 loss=0.2332 acc=0.9313 lr=0.000050 step/sec=4.55 | ETA 00:03:54
    [2021-04-14 01:03:23,199] [   TRAIN] - Epoch=2/3, Step=230/282 loss=0.1559 acc=0.9594 lr=0.000050 step/sec=4.57 | ETA 00:03:53
    [2021-04-14 01:03:25,389] [   TRAIN] - Epoch=2/3, Step=240/282 loss=0.1173 acc=0.9625 lr=0.000050 step/sec=4.57 | ETA 00:03:52
    [2021-04-14 01:03:27,578] [   TRAIN] - Epoch=2/3, Step=250/282 loss=0.1320 acc=0.9594 lr=0.000050 step/sec=4.57 | ETA 00:03:51
    [2021-04-14 01:03:29,774] [   TRAIN] - Epoch=2/3, Step=260/282 loss=0.1583 acc=0.9594 lr=0.000050 step/sec=4.55 | ETA 00:03:50
    [2021-04-14 01:03:31,971] [   TRAIN] - Epoch=2/3, Step=270/282 loss=0.1585 acc=0.9469 lr=0.000050 step/sec=4.55 | ETA 00:03:49
    [2021-04-14 01:03:34,166] [   TRAIN] - Epoch=2/3, Step=280/282 loss=0.2247 acc=0.9406 lr=0.000050 step/sec=4.56 | ETA 00:03:48
    [2021-04-14 01:03:37,962] [    EVAL] - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - [Evaluation result] avg_acc=0.9121
    [2021-04-14 01:03:51,592] [    EVAL] - Saving best model to ckpt/best_model [best acc=0.9121]
    [2021-04-14 01:03:51,595] [    INFO] - Saving model checkpoint to ckpt/epoch_2
    [2021-04-14 01:04:06,712] [   TRAIN] - Epoch=3/3, Step=10/282 loss=0.1233 acc=0.9656 lr=0.000050 step/sec=0.37 | ETA 00:04:32
    [2021-04-14 01:04:08,883] [   TRAIN] - Epoch=3/3, Step=20/282 loss=0.1088 acc=0.9656 lr=0.000050 step/sec=4.61 | ETA 00:04:30
    [2021-04-14 01:04:11,058] [   TRAIN] - Epoch=3/3, Step=30/282 loss=0.0930 acc=0.9781 lr=0.000050 step/sec=4.60 | ETA 00:04:29
    [2021-04-14 01:04:13,215] [   TRAIN] - Epoch=3/3, Step=40/282 loss=0.0765 acc=0.9844 lr=0.000050 step/sec=4.64 | ETA 00:04:27
    [2021-04-14 01:04:15,377] [   TRAIN] - Epoch=3/3, Step=50/282 loss=0.1030 acc=0.9656 lr=0.000050 step/sec=4.62 | ETA 00:04:26
    [2021-04-14 01:04:17,545] [   TRAIN] - Epoch=3/3, Step=60/282 loss=0.1028 acc=0.9688 lr=0.000050 step/sec=4.61 | ETA 00:04:25
    [2021-04-14 01:04:19,717] [   TRAIN] - Epoch=3/3, Step=70/282 loss=0.0757 acc=0.9812 lr=0.000050 step/sec=4.60 | ETA 00:04:23
    [2021-04-14 01:04:21,892] [   TRAIN] - Epoch=3/3, Step=80/282 loss=0.1002 acc=0.9781 lr=0.000050 step/sec=4.60 | ETA 00:04:22
    [2021-04-14 01:04:24,066] [   TRAIN] - Epoch=3/3, Step=90/282 loss=0.0679 acc=0.9844 lr=0.000050 step/sec=4.60 | ETA 00:04:21
    [2021-04-14 01:04:26,238] [   TRAIN] - Epoch=3/3, Step=100/282 loss=0.0557 acc=0.9906 lr=0.000050 step/sec=4.60 | ETA 00:04:20
    [2021-04-14 01:04:28,418] [   TRAIN] - Epoch=3/3, Step=110/282 loss=0.1407 acc=0.9625 lr=0.000050 step/sec=4.59 | ETA 00:04:18
    [2021-04-14 01:04:30,609] [   TRAIN] - Epoch=3/3, Step=120/282 loss=0.0919 acc=0.9750 lr=0.000050 step/sec=4.56 | ETA 00:04:17
    [2021-04-14 01:04:32,804] [   TRAIN] - Epoch=3/3, Step=130/282 loss=0.0671 acc=0.9844 lr=0.000050 step/sec=4.56 | ETA 00:04:16
    [2021-04-14 01:04:34,995] [   TRAIN] - Epoch=3/3, Step=140/282 loss=0.0743 acc=0.9781 lr=0.000050 step/sec=4.56 | ETA 00:04:15
    [2021-04-14 01:04:37,187] [   TRAIN] - Epoch=3/3, Step=150/282 loss=0.0501 acc=0.9844 lr=0.000050 step/sec=4.56 | ETA 00:04:14
    [2021-04-14 01:04:39,383] [   TRAIN] - Epoch=3/3, Step=160/282 loss=0.0634 acc=0.9844 lr=0.000050 step/sec=4.55 | ETA 00:04:13
    [2021-04-14 01:04:41,578] [   TRAIN] - Epoch=3/3, Step=170/282 loss=0.0502 acc=0.9875 lr=0.000050 step/sec=4.56 | ETA 00:04:12
    [2021-04-14 01:04:43,778] [   TRAIN] - Epoch=3/3, Step=180/282 loss=0.1112 acc=0.9688 lr=0.000050 step/sec=4.55 | ETA 00:04:12
    [2021-04-14 01:04:45,970] [   TRAIN] - Epoch=3/3, Step=190/282 loss=0.0790 acc=0.9781 lr=0.000050 step/sec=4.56 | ETA 00:04:11
    [2021-04-14 01:04:48,167] [   TRAIN] - Epoch=3/3, Step=200/282 loss=0.1113 acc=0.9656 lr=0.000050 step/sec=4.55 | ETA 00:04:10
    [2021-04-14 01:04:50,363] [   TRAIN] - Epoch=3/3, Step=210/282 loss=0.0524 acc=0.9875 lr=0.000050 step/sec=4.55 | ETA 00:04:09
    [2021-04-14 01:04:52,563] [   TRAIN] - Epoch=3/3, Step=220/282 loss=0.0905 acc=0.9625 lr=0.000050 step/sec=4.55 | ETA 00:04:08
    [2021-04-14 01:04:54,764] [   TRAIN] - Epoch=3/3, Step=230/282 loss=0.0931 acc=0.9719 lr=0.000050 step/sec=4.54 | ETA 00:04:07
    [2021-04-14 01:04:56,963] [   TRAIN] - Epoch=3/3, Step=240/282 loss=0.1092 acc=0.9750 lr=0.000050 step/sec=4.55 | ETA 00:04:07
    [2021-04-14 01:04:59,156] [   TRAIN] - Epoch=3/3, Step=250/282 loss=0.0717 acc=0.9844 lr=0.000050 step/sec=4.56 | ETA 00:04:06
    [2021-04-14 01:05:01,353] [   TRAIN] - Epoch=3/3, Step=260/282 loss=0.0827 acc=0.9781 lr=0.000050 step/sec=4.55 | ETA 00:04:05
    [2021-04-14 01:05:03,555] [   TRAIN] - Epoch=3/3, Step=270/282 loss=0.1009 acc=0.9750 lr=0.000050 step/sec=4.54 | ETA 00:04:04
    [2021-04-14 01:05:05,752] [   TRAIN] - Epoch=3/3, Step=280/282 loss=0.1048 acc=0.9781 lr=0.000050 step/sec=4.55 | ETA 00:04:04
    [2021-04-14 01:05:09,544] [    EVAL] - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - Evaluation on validation dataset: / - Evaluation on validation dataset: - - Evaluation on validation dataset: \ - Evaluation on validation dataset: | - [Evaluation result] avg_acc=0.9107
    [2021-04-14 01:05:09,547] [    INFO] - Saving model checkpoint to ckpt/epoch_3


## 预测


```python
# 预测
data = [
    # 房产
    ["昌平京基鹭府10月29日推别墅1200万套起享97折　　新浪房产讯(编辑郭彪)京基鹭府(论坛相册户型样板间点评地图搜索)售楼处位于昌平区京承高速北七家出口向西南公里路南。项目预计10月29日开盘，总价1200万元/套起，2012年年底入住。待售户型为联排户型面积为410-522平方米，独栋户型面积为938平方米，双拼户型面积为522平方米。　　京基鹭府项目位于昌平定泗路与东北路交界处。项目周边配套齐全，幼儿园：伊顿双语幼儿园、温莎双语幼儿园；中学：北师大亚太实验学校、潞河中学(北京市重点)；大学：王府语言学校、北京邮电大学、现代音乐学院；医院：王府中西医结合医院(三级甲等)、潞河医院、解放军263医院、安贞医院昌平分院；购物：龙德广场、中联万家商厦、世纪华联超市、瑰宝购物中心、家乐福超市；酒店：拉斐特城堡、鲍鱼岛；休闲娱乐设施：九华山庄、温都温泉度假村、小汤山疗养院、龙脉温泉度假村、小汤山文化广场、皇港高尔夫、高地高尔夫、北鸿高尔夫球场；银行：工商银行、建设银行、中国银行、北京农村商业银行；邮局：中国邮政储蓄；其它：北七家建材城、百安居建材超市、北七家镇武装部、北京宏翔鸿企业孵化基地等，享受便捷生活。"],
    # 游戏
    ["尽管官方到今天也没有公布《使命召唤：现代战争2》的游戏详情，但《使命召唤：现代战争2》首部包含游戏画面的影片终于现身。虽然影片仅有短短不到20秒，但影片最后承诺大家将于美国时间5月24日NBA职业篮球东区决赛时将会揭露更多的游戏内容。　　这部只有18秒的广告片闪现了9个镜头，能够辨识的场景有直升机飞向海岛军事工事，有飞机场争夺战，有潜艇和水下工兵，有冰上乘具，以及其他的一些镜头。整体来看《现代战争2》很大可能仍旧与俄罗斯有关。　　片尾有一则预告：“May24th，EasternConferenceFinals”，这是什么？这是说当前美国NBA联赛东部总决赛的日期。原来这部视频是NBA季后赛奥兰多魔术对波士顿凯尔特人队时，TNT电视台播放的广告。"],
    # 体育
    ["罗马锋王竟公然挑战两大旗帜拉涅利的球队到底错在哪　　记者张恺报道主场一球小胜副班长巴里无可吹捧，罗马占优也纯属正常，倒是托蒂罚失点球和前两号门将先后受伤(多尼以三号身份出场)更让人揪心。阵容规模扩大，反而表现不如上赛季，缺乏一流强队的色彩，这是所有球迷对罗马的印象。　　拉涅利说：“去年我们带着嫉妒之心看国米，今年我们也有了和国米同等的超级阵容，许多教练都想有罗马的球员。阵容广了，寻找队内平衡就难了，某些时段球员的互相排斥和跟从前相比的落差都正常。有好的一面，也有不好的一面，所幸，我们一直在说一支伟大的罗马，必胜的信念和够级别的阵容，我们有了。”拉涅利的总结由近一阶段困扰罗马的队内摩擦、个别球员闹意见要走人而发，本赛季技术层面强化的罗马一直没有上赛季反扑的面貌，内部变化值得球迷关注。"],
    # 教育
    ["新总督致力提高加拿大公立教育质量　　滑铁卢大学校长约翰斯顿先生于10月1日担任加拿大总督职务。约翰斯顿先生还曾任麦吉尔大学长，并曾在多伦多大学、女王大学和西安大略大学担任教学职位。　　约翰斯顿先生在就职演说中表示，要将加拿大建设成为一个“聪明与关爱的国度”。为实现这一目标，他提出三个支柱：支持并关爱家庭、儿童；鼓励学习与创造；提倡慈善和志愿者精神。他尤其强调要关爱并尊重教师，并通过公立教育使每个人的才智得到充分发展。"]
]

label_list=['体育', '科技', '社会', '娱乐', '股票', '房产', '教育', '时政', '财经', '星座', '游戏', '家居', '彩票', '时尚']
label_map = { 
    idx: label_text for idx, label_text in enumerate(label_list)
}

model = hub.Module(
    name='ernie',
    task='seq-cls',
    load_checkpoint='./ckpt/best_model/model.pdparams',
    label_map=label_map)
results = model.predict(data, max_seq_len=128, batch_size=1, use_gpu=True)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text[0], results[idx]))
```

    [2021-04-14 01:05:30,431] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-1.0/ernie_v1_chn_base.pdparams
    [2021-04-14 01:05:35,372] [    INFO] - Loaded parameters from /home/aistudio/ckpt/best_model/model.pdparams
    [2021-04-14 01:05:35,487] [    INFO] - Found /home/aistudio/.paddlenlp/models/ernie-1.0/vocab.txt


    Data: 昌平京基鹭府10月29日推别墅1200万套起享97折　　新浪房产讯(编辑郭彪)京基鹭府(论坛相册户型样板间点评地图搜索)售楼处位于昌平区京承高速北七家出口向西南公里路南。项目预计10月29日开盘，总价1200万元/套起，2012年年底入住。待售户型为联排户型面积为410-522平方米，独栋户型面积为938平方米，双拼户型面积为522平方米。　　京基鹭府项目位于昌平定泗路与东北路交界处。项目周边配套齐全，幼儿园：伊顿双语幼儿园、温莎双语幼儿园；中学：北师大亚太实验学校、潞河中学(北京市重点)；大学：王府语言学校、北京邮电大学、现代音乐学院；医院：王府中西医结合医院(三级甲等)、潞河医院、解放军263医院、安贞医院昌平分院；购物：龙德广场、中联万家商厦、世纪华联超市、瑰宝购物中心、家乐福超市；酒店：拉斐特城堡、鲍鱼岛；休闲娱乐设施：九华山庄、温都温泉度假村、小汤山疗养院、龙脉温泉度假村、小汤山文化广场、皇港高尔夫、高地高尔夫、北鸿高尔夫球场；银行：工商银行、建设银行、中国银行、北京农村商业银行；邮局：中国邮政储蓄；其它：北七家建材城、百安居建材超市、北七家镇武装部、北京宏翔鸿企业孵化基地等，享受便捷生活。 	 Lable: 房产
    Data: 尽管官方到今天也没有公布《使命召唤：现代战争2》的游戏详情，但《使命召唤：现代战争2》首部包含游戏画面的影片终于现身。虽然影片仅有短短不到20秒，但影片最后承诺大家将于美国时间5月24日NBA职业篮球东区决赛时将会揭露更多的游戏内容。　　这部只有18秒的广告片闪现了9个镜头，能够辨识的场景有直升机飞向海岛军事工事，有飞机场争夺战，有潜艇和水下工兵，有冰上乘具，以及其他的一些镜头。整体来看《现代战争2》很大可能仍旧与俄罗斯有关。　　片尾有一则预告：“May24th，EasternConferenceFinals”，这是什么？这是说当前美国NBA联赛东部总决赛的日期。原来这部视频是NBA季后赛奥兰多魔术对波士顿凯尔特人队时，TNT电视台播放的广告。 	 Lable: 游戏
    Data: 罗马锋王竟公然挑战两大旗帜拉涅利的球队到底错在哪　　记者张恺报道主场一球小胜副班长巴里无可吹捧，罗马占优也纯属正常，倒是托蒂罚失点球和前两号门将先后受伤(多尼以三号身份出场)更让人揪心。阵容规模扩大，反而表现不如上赛季，缺乏一流强队的色彩，这是所有球迷对罗马的印象。　　拉涅利说：“去年我们带着嫉妒之心看国米，今年我们也有了和国米同等的超级阵容，许多教练都想有罗马的球员。阵容广了，寻找队内平衡就难了，某些时段球员的互相排斥和跟从前相比的落差都正常。有好的一面，也有不好的一面，所幸，我们一直在说一支伟大的罗马，必胜的信念和够级别的阵容，我们有了。”拉涅利的总结由近一阶段困扰罗马的队内摩擦、个别球员闹意见要走人而发，本赛季技术层面强化的罗马一直没有上赛季反扑的面貌，内部变化值得球迷关注。 	 Lable: 体育
    Data: 新总督致力提高加拿大公立教育质量　　滑铁卢大学校长约翰斯顿先生于10月1日担任加拿大总督职务。约翰斯顿先生还曾任麦吉尔大学长，并曾在多伦多大学、女王大学和西安大略大学担任教学职位。　　约翰斯顿先生在就职演说中表示，要将加拿大建设成为一个“聪明与关爱的国度”。为实现这一目标，他提出三个支柱：支持并关爱家庭、儿童；鼓励学习与创造；提倡慈善和志愿者精神。他尤其强调要关爱并尊重教师，并通过公立教育使每个人的才智得到充分发展。 	 Lable: 时政

