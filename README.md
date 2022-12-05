В данной работе представлена реализация нейронной сети VGG16 и оптимизатора AdaBound с использованием библиотеки numpy, а также сравнение оптимизаторов AdaBound и классического Adam на датасете Stanford cars с использованием предобученной сети VGG16. 

Компьютерное зрение – одна из главных областей теории искусственных интеллектов, активно
развивающейся последние 50 лет с наибольшим пиком активности последние 10 лет.
Глубинное (глубокое) обучение – раздел теории нейронных сетей, разрабатывающий и исследующий
модели нейронных сетей с большим количеством слоев.
Формально, нейросети решают задачу аппроксимации неизвестной функции. На сегодняшний день все
объяснения эффективности методов обучения нейросетей являются гипотетическими. Часть гипотез
подтверждается многочисленными экспериментами, часть гипотез опровергается, что приводит к
возникновению новых гипотез
Нейронные сети учатся послойно извлекать из изображений информацию, уровень сложности
которой растет с каждым слоем. Например, первый слой запоминает штрихи, перепады цветов и яркости,
второй слой исследует комбинации выходов первого слоя и так далее. Эксперименты показывают, что на
более глубоких слоях производится запоминание таких признаков, как, например, колесо автомобиля, форма
глаз и т.д
Сверточной нейронной сетью называют нейросеть со следующей архитектурой, удобно
визуализируемой в трехмерном пространстве (Рис. 1):
![image](https://user-images.githubusercontent.com/58371161/203812542-82bbb35e-b04a-4e35-b377-a42f507cff95.png)

Рис.1 - сверточная нейронная сеть

С математической точки зрения, такая архитектура позволяет уменьшить количество параметров сети,
что позволяет улучшить обобщающие свойства сети. Кроме того, такая архитектура позволяет извлекать
локальные свойства из входных данных.
В качестве входа каждый слой сети принимает трехмерный тензор, на практике чаще всего
являющийся изображением. Таким образом, выходом данного блока будет новое изображение, являющееся
результатом операции, называющейся в графическом редактировании применением фильтра к входу.
Применение фильтра называют сверткой.
Процесс обучения и свертки интерпретируются следующим образом:
1. Фильтры первого запоминают общие свойства, характерные для входных данных.
2. Фильтры последующих слоев запоминают характерные комбинации свойств, полученных
предшествующими слоями
3. С возрастанием глубины слои извлекают все более глубокие связи между свойствами,
характерными для каждого класса входных данных. 
Выходом сверточной сети является большой набор преобразованных данных, содержащих свойства
входного изображения, поэтому после CNN ставятся полносвязные слои, которые производят оценку и интерпретацию извлеченных
свойств, а последний слой выдает вероятности классов.

VGG16 — модель сверточной нейронной сети, предложенная K. Simonyan и A. Zisserman из Оксфордского университета в статье “Very Deep Convolutional Networks for Large-Scale Image Recognition”. Модель достигает точности 92.7% — топ-5, при тестировании на ImageNet в задаче распознавания объектов на изображении.
Архитектура VGG16 представлена на рисунке ниже.

Архитектура нейросети vgg16 (Рис. 2)
![image](https://user-images.githubusercontent.com/58371161/205052117-76b502ce-c2b2-400c-a055-ce1a6a84a656.png)
                                                                    Рис. 2 - Архитектура vgg16

На вход слоя conv1 подаются RGB изображения размера 224х224. Далее изображения проходят через стек сверточных слоев, в которых используются фильтры с очень маленьким рецептивным полем размера 3х3.
Пространственное дополнение (padding) входа сверточного слоя выбирается таким образом, чтобы пространственное разрешение сохранялось после свертки, то есть дополнение равно 1 для 3х3 сверточных слоев. Пространственный пулинг осуществляется при помощи пяти max-pooling слоев, которые следуют за одним из сверточных слоев . Операция max-pooling выполняется на окне размера 2х2 пикселей с шагом 2.

После стека сверточных слоев идут три полносвязных слоя: первые два имеют по 4096 каналов, третий — 1000 каналов. Последним идет soft-max слой. Конфигурация полносвязных слоев одна и та же во всех нейросетях.Все скрытые слои снабжены ReLU.
Конфигурация архитектур vgg представлена на рисунке 3

![image](https://user-images.githubusercontent.com/58371161/205052293-472953a5-e7c8-476d-ad43-6a36d88fc112.png)

Рис. 3 - конфигурация сетей vgg

К сожалению, сеть VGG имеет два серьезных недостатка:

Очень медленная скорость обучения.
Сама архитектура сети весит слишком много (появляются проблемы с диском и пропускной способностью)
Из-за глубины и количества полносвязных узлов, VGG16 весит более 533 МБ. Это делает процесс развертывания VGG утомительной задачей.

При обучении одну из важнейших ролей играют оптимизаторы - способы изменения и настройки весов нейронных сетей. Классический инструмент - Adam, ниже представлены принципы, по которым происходит изменение параметров сети:
![image](https://user-images.githubusercontent.com/58371161/205631563-4b03758a-24ad-4bd0-bcc5-64820ec6840a.png)
Adam — один из самых эффективных алгоритмов оптимизации в обучении нейронных сетей. Он сочетает в себе идеи RMSProp и оптимизатора импульса. Вместо того чтобы адаптировать скорость обучения параметров на основе среднего первого момента (среднего значения), как в RMSProp, Adam также использует среднее значение вторых моментов градиентов. В частности, алгоритм вычисляет экспоненциальное скользящее среднее градиента и квадратичный градиент, а параметры beta1 и beta2 управляют скоростью затухания этих скользящих средних
Оптимизатор Adabound настраивает веса согласно следующим принципам: 
![image](https://user-images.githubusercontent.com/58371161/205635233-f7ae3da4-3c89-42be-81ca-d302637e62c8.png)
![image](https://user-images.githubusercontent.com/58371161/205635451-1d17c8dc-23d3-45fd-aaf5-12451d7cef4f.png)


Методы адаптивной оптимизации, такие как AdaGrad, RMSprop и Adam, хорошо показывают себя для достижения быстрого процесса обучения с масштабированием learning rate, используя перывый и второй моментум, как было показано в adam. Несмотря на то, что они преобладают, наблюдается, что они плохо обобщаются по сравнению с SGD или даже не сходятся из-за нестабильной и экстремальной скорости обучения. Adabound использует динамические ограничения скорости обучения для достижения постепенного и плавного перехода от адаптивных методов к SGD. Экспериментальные результаты [2] показывают, что новые варианты могут устранить разрыв в обобщении между адаптивными методами и SGD и в то же время поддерживать более высокую скорость обучения на ранних этапах обучения. Более того, они могут значительно улучшить свои прототипы, особенно в сложных глубоких сетях.

Результаты обучения VGG на Stanford cars с Adam:
![image](https://user-images.githubusercontent.com/58371161/205674883-97da1487-9374-4f02-894e-6b05da9fae2a.png)





