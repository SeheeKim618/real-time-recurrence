from jax import random, ops
import jax.numpy as np
import matplotlib.pyplot as plt

"""
Sequence to sequence dataset. 
Input should be repeated in output.
"""

class CopyTaskData:

    def __init__(self, batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue, bias=False):
        self.batch_size = batch_size
        self.maxSeqLength = maxSeqLength
        self.minSeqLength = np.maximum(1, (maxSeqLength-5)) #minSeqLength
        self.bits = bits
        self.padding = padding
        self.lowValue = lowValue
        self.highValue = highValue 
        self.bias = bias

        self.input = self.bits+2 # start and stop
        self.output = self.bits

    def calcSample(self, key):
        #print("maxSeqLength at Data: ", self.maxSeqLength)
        #print("minSeqLength at Data: ", self.minSeqLength)
        seqLength = random.randint(key, (1,), minval=self.minSeqLength, maxval=self.maxSeqLength+1)[0]
        pattern = random.choice(key, np.array([self.lowValue, self.highValue]), shape=(seqLength, self.output))

        x = np.ones(((self.maxSeqLength*2)+2,self.input)) * self.padding
        y = np.ones(((self.maxSeqLength*2)+2,self.output)) * self.padding

        startSeq = np.ones((self.input)) * self.padding
        startSeq = startSeq.at[0].set(1.0)

        endSeq = np.ones((self.input)) * self.padding
        endSeq = endSeq.at[1].set(1.0)

        #print(x.shape)
        #print(pattern.shape)

        x = x.at[0].set(startSeq)
        x = x.at[1:(1+seqLength),2:].set(pattern)
        x = x.at[(1+seqLength)].set(endSeq)

        y = y.at[seqLength+2:(2*seqLength)+2,:].set(pattern)

        return x, y

    def getSample(self, key):
        
        inputs = []
        outputs = []

        for i, k in zip(range(self.batch_size), random.split(key, self.batch_size)):

            x, y = self.calcSample(k)

            if (self.bias):
                x = np.append(x, np.ones((x.shape[0],1)), axis=1)

            inputs.append(x)
            outputs.append(y)

        return np.array(inputs), np.array(outputs)

    def __iter__(self):
        # 이 메서드는 데이터셋을 반복하기 위한 반복자를 반환합니다.
        # 여기서는 간단한 예시로, 클래스 인스턴스 자체를 반복자로 사용합니다.
        # 이를 위해 반복 상태를 추적하는 속성을 초기화합니다.
        self.iter_count = 0
        return self

    def __next__(self):
        # 반복자 프로토콜의 일부로, __next__ 메서드는 다음 요소를 반환합니다.
        if self.iter_count < self.batch_size:
            # batch_size만큼 반복하면서 샘플을 생성합니다.
            self.iter_count += 1
            x, y = self.getSample(self.key)
            return x, y
        else:
            # 지정된 반복 횟수를 초과하면 반복을 종료합니다.
            raise StopIteration

if __name__ == "__main__":

    key = random.PRNGKey(1)
    
    maxSeqLength = 10
    minSeqLength = 7
    bits = 1
    padding = 0
    lowValue = 0
    highValue = 1
    batch_size= 1

    data = CopyDataSet(batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue)
    x, y = data.getSample(key)

    x = x[0]
    y = y[0]

    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.subplots_adjust(top=0.85,bottom=0.15,left=0.05,right=0.95)

    cmap = plt.get_cmap('jet')
    t=ax1.matshow(x.T,aspect='auto',cmap=cmap)
    ax1.set_ylabel("Input")
    p=ax2.matshow(y.T,aspect='auto',cmap=cmap)
    ax2.set_ylabel("Traget")

    fig.suptitle('Copy Task')
    fig.colorbar(t,ax=(ax1,ax2),orientation="vertical",fraction=0.1)

    plt.savefig('copy_task.png')
