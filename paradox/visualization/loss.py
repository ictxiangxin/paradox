import matplotlib.pyplot as plot


class VisualLoss:
    def __init__(self, title: str=None):
        self.__title = title
        self.__loss_list = []
        plot.figure(self.__title)

    def add(self, loss: float):
        self.__loss_list.append(loss)

    def show(self):
        if self.__title is not None:
            plot.title(self.__title)
        x = range(len(self.__loss_list))
        plot.plot(x, self.__loss_list, 'r', label='Loss')
        plot.legend()
        plot.figure().show()
