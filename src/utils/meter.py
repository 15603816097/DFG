class AverageMeter:


    def __init__(self):

        self.reset()



    def reset(self):

        self.sum=0.0

        self.count=0



    def update(
        self,
        value,
        n=1
    ):

        self.sum += value*n

        self.count += n



    @property
    def avg(self):

        if self.count==0:

            return 0

        return self.sum/self.count
