import numpy as np
import pandas as pd
from defaultlist import defaultlist
import matplotlib.pyplot as plt

class MLP:
    """
    Модель многослойного перцептрона, используемая для
    многоклассовой классификации. Особенности:
    - Start weights: Kaiming He initialization;
    - Gradient descent optimisation: RMSProp
    - ReLu activation
    - CONFIGURABLE network architecture

    """
    def __init__(self,
                 hidden_net: tuple|list = (10),  # Structure of hidden layers
                 learning_rate: float = 0.01,  # Speed
                 max_epochs: int = 100,  # Количество эпох
                 one_batch = 50,  # Размер олного батча
                 random_state=42,
                 gamma=0.9,
                 epsilon=10 ** (-3),
                 start_loc=0.0,
                 start_scale=0.1
                 ):
        self. hidden_net = hidden_net
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = one_batch
        self.random_state = random_state
        self.gamma = gamma  # сглаживающий параметр для RSMProp
        self.epsilon = epsilon  # сглаживающий параметр для RSMProp
        self.loc = start_loc
        self.scale = start_scale

    def __create_start_weight(self, dimension: tuple[int],
                              loc, scale) -> np.array:
        """
        Инициализация весов в начале обучения.
        dimension --> W = [ w0, w1, w2, w3, w4 ],
        где w0 (OR b) - член смещения
    """

        rgen = np.random.RandomState(self.random_state)
        # Начальные веса и член смещения - в одном векторе
        return  rgen.normal(loc=loc, scale=scale, size=dimension)

    @staticmethod
    def __show_architecture(layers):
        return (f'- входной слой: {layers[0]}',
            *[f"скрытый слой шириной {h}" for h in layers[1:-1]],
              f'выходной слой {layers[-1]}')

    def __build_net_architecture(self, input_dim: int, output_dim: int) -> None:
        """
        Function takes info about hidden layers and helps program to create it
        Мы подаём на вход список: его длина - количество скрытых слоёв, а значение каждого элемента - ширина соответствующего слоя.
        Для каждого слоя генерируем начальные веса - матрицу соответствующей размерности
        """
        self.hidden_net = list(self.hidden_net)  # In case the tuple was given
        layers = [input_dim, *self.hidden_net, output_dim]

        print(*self.__show_architecture(layers), sep='\n- ')  # length and depth
        weights = []
        # Размерность матрицы весов определяется шириной слоя, который подаётся на вход (это количество столбцов),
        # и шириной следующего слоя (это количество строк)
        for input_dim, output_dim in zip(
                layers[:-1], layers[1:]  # Получаем пары (слой, следующий слой)

        ):
            # К размерности входного слоя мы прибавляем единицу - так мы учитываем ФИКТИВНЫЙ столбец
            weights.append(
                # Используем метод "Kaiming He":  нормальное распределение мы умножаем на особый множитель,
                 # который зависит от ширины конкретного слоя
                self.__create_start_weight((output_dim, input_dim + 1),
                    loc=self.loc, scale=self.scale) * np.sqrt(2 / input_dim)  # "He-et-al" initialization
            )
        # В итоге мы имеем m+1 весов для m скрытых слоёв.
        self.W = weights

    @staticmethod
    def __extend_matrix(matrix) -> pd.DataFrame:
        """
        Функция для добавления фиктивного единичного вектора
        к матрице
        """
        # Будем работать с локальной копией, чтобы не менять исходный датафрейм
        matrix = matrix.copy()
        matrix.insert(0, 'fictive', 1)
        return matrix

    @staticmethod
    def convert_target(y) -> pd.DataFrame:
        """
        Функция для преобразования истинных меток методом One-hot Encoding. Принимает
        вектор выходных переменных и возвращает матрицу, где стобцы - все уникальные значения.
        Вообще, этот метод подразумевает k-1 столбцов для k уникальных значений во избежание мультиколлинеарности.
        Однако же, мы просто хотим расширить наш вектор Y до матрицы, чтобы можно было искать градиент, поэтому drop_first=False.
        """
        y = y.copy()
        return pd.get_dummies(y, drop_first=False, dummy_na=False, dtype='float')

    @staticmethod
    def __relu(T) -> pd.DataFrame:
        """
        Функция активации RELU.
        T --> output H(T)
        """
        return np.maximum(T, 0)

    @staticmethod
    def __relu_deriv(T) -> pd.DataFrame:
        """
        Производная функции активации, необходима для поиска градиента.
        Сравниваем все значения датафрейма T с нулём - получаем матрицу с True и False, эти значения переводим в бинарные значения
        """
        return (T >= 0).astype(float, copy=True)

    @staticmethod
    def __softmax(array) -> pd.DataFrame:
        """ Converts a vector of K real numbers into a probability distribution of K possible outcomes"""
        P = []
        for item in array.iterrows():  # Item - одна строка датафрейма формата <(порядковый номер строки, Series)>
            e = np.exp(item[1])  # Получаем вектор всех значений для одного наблюдения
            P.append(e / np.sum(e))  # Делим вектор на сумму экспонент, получая вероятностное распределение
        return pd.DataFrame(P)

    @staticmethod
    def __CE_log_loss(y, p):
        """
        Логарифмическая функция потерь - кросс-энтропия - для небинарной классификации:
        Получаем на вход две матрицы KxN - K классов (уникальных меток) на N образцов.
        Логарифмируем предсказанные вероятности, после перемножаем матрицы и получаем матрицу KxK;
        складываем все K^2 значений. Ставим минус, чтобы преобразовать функцию правдоподобия в функцию потерь

        """
        return - np.array([np.log(p[:, i]).T @ y[:, i] for i in range(y.shape[1])]).sum()

    def __forward_prop(self, X) -> tuple[defaultlist, defaultlist, pd.DataFrame]:
        """ Расчёт выходного слоя"""
        T = defaultlist()  # Линейная часть расчёта
        H = defaultlist()  # Нелинейная часть (RELU)
        H[0] = self.__extend_matrix(X)
        # Проходим по всем слоям
        for layer in range(1, len(self.W) + 1):
            T[layer] = H[layer - 1] @ self.W[layer - 1].T
            H[layer] = self.__extend_matrix(self.__relu(T[layer]))
        # На последнем слое вместо функции активации используем софтмакс
        P = self.__softmax(T[-1])
        return T, H, P

    def __RMSProp(self, grad, i) -> np.array:
        """
        Функция для оптимизации градиентного спуска методом
        Root Mean Square Propagation.
        Возвращает матрицу deltaW.
        """
        # Root Mean Square Propagation
        self.E_[i] = self.gamma * self.E_[i] + (1 - self.gamma) * grad ** 2
        return grad / (np.sqrt(self.E_[i] + self.epsilon))

    def __backward_prop(self, T, H, E) -> list[np.array]:
        """ Обратное распространение ошибки"""
        dH = defaultlist()
        dW = defaultlist()
        dT = defaultlist()
        dT[len(self.W)] = (E).to_numpy()
        for layer in range(len(self.W) - 1, 0, -1):
            dW[layer] = dT[layer + 1].T @ H[layer].to_numpy()
            dH[layer] = dT[layer + 1] @ self.W[layer][:, 1:]  # Вектор смещения не учитываем, чтобы совпадала размерность
            rd = self.__relu_deriv(T[layer]).to_numpy()
            dT[layer] = dH[layer] * rd
        dW[0] = dT[1].T @ H[0].to_numpy()
        return dW


    def fit(self, X, Y):
        """ Функция для обучения модели"""
        self.loss_function = []
        # X = copy.deepcopy(X)
        # Определим размерность входного и выходного слоёв
        output_dim, input_dim = Y.shape[1], X.shape[1]
        self.__build_net_architecture(input_dim, output_dim)
        # Инициализируем вектор бегущего среднего для RMSP - это
        # нулевой вектор, длина которого - число разных весов..
        self.E_ = [0] * (len(self.W) + 1)
        for epoch in range(self.max_epochs):
            # Prediction
            X = X.sample(frac=1, random_state=self.random_state)
            Y = Y.sample(frac=1, random_state=self.random_state)
            # Используем пакетный градиентный спуск
            for k in range(
                # Если размер батча слишком большой (например, 160 при размере датасета в 150),
                # мы должны избежать нулевой итерации по range(0, 0)
                    max(1, X.shape[0] // self.batch_size) # Количество батчей
            ):
                row = k * self.batch_size  # Last batch row
                X_batch = X.iloc[row: row + self.batch_size]
                Y_batch = Y.iloc[row: row+ self.batch_size]

                T, H, P = self.__forward_prop(X_batch)
                # Функция издержек
                self.loss_function.append(self.__CE_log_loss(Y_batch.to_numpy(), P.to_numpy()))
                # dE_dW1 = np.nan_to_num(dE_dW1)
                # dE_dW2 = np.nan_to_num(dE_dW2)

                dW = self.__backward_prop(
                    T, H,
                    P-Y_batch  # Матрица ошибки Е при данных весах
                )
                # Обновление весов
                for i in range(0, len(self.W)):
                    rate = self.learning_rate * self.__RMSProp(dW[i], i)
                    self.W[i] -= rate

        print(f'Model has been trained successfully!')

    def predict(self, X):
        """
        Рассчитываем выходной слой, отбрасываем лишнюю
        информацию и сортируем
        """
        return self.__forward_prop(X)[2].sort_index()

def show_loss(loss) -> None:
    """ Функция для отображения графика функции потерь"""
    plt.plot(loss)
    plt.title("LOSS")
    plt.show()


def calculate_scores(pred, target) -> str:
    """ Find the accuracy"""
    scores = 0
    for check, true in zip(pred, target):
        if float(check) == true:
            scores +=1
    return f'{scores/len(target)*100}%'

def main():
    """ Классифицируем датасет с ирисами"""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    omg = load_iris()
    df = pd.DataFrame(data=np.c_[omg.data, omg.target],
                      columns=([i[:-4] for i in omg['feature_names']] + ['target']))
    Y = df.target
    Y_converted  = MLP.convert_target(Y)  # Преобразование вектора в матрицу из K столбцов
    X = df.iloc[:, [0, 1, 2, 3]]
    # model = MLP(hidden_net=[5, 5, 5], learning_rate=0.001, max_epochs=1000, one_batch=20, random_state=10, start_loc=0.0)
    model = MLP(hidden_net=[12, 9], learning_rate=0.0005, max_epochs=1600, one_batch=150, random_state=20)
    # model = MLP(hidden_net=[10], learning_rate=0.0001, max_epochs=1500, one_batch=20, random_state=99)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_converted, test_size=0.2, random_state=model.random_state)

    model.fit(X_train, Y_train)
    P = model.predict(X)
    print("Model's accuracy is:  ", calculate_scores(P.to_numpy().argmax(axis=1).T, Y))
    show_loss(model.loss_function)

if __name__ == '__main__':
    main()
