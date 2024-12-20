Для выполнения практических заданий используются суперкомпьютерные вычислительные ресурсы факультета ВМК.
http://hpc.cs.msu.ru/
В каждой задаче требуется:
1) Для предложенного алгоритма реализовать несколько версий параллельных программ с использованием технологии OpenMP.
a) Вариант параллельной программы с распределением витков циклов при помощи директивы for.
б) Вариант параллельной программы с использованием механизма задач (директива task).
б') Для программ с регулярной зависимостью по данным вместо механизма задач допускается реализация и сравнение различных версий конвейерного выполнения циклов, параллелизм по гиперплоскостям и др.

2) Реализовать параллельную версию программы с использованием технологии MPI.

3) Убедиться в корректности разработанных версий программ.

4) Начальные параметры для задачи должны быть подобраны таким образом, чтобы:
a) Задача помещалась в оперативную память одного узла кластера.
б) Время решения задачи было в примерном диапазоне 5 сек.-15 минут.

5) Исследовать эффективность полученных параллельных программ на суперкомпьютере Polus.
a) Сравнить варианты разработанных версий параллельных программ.
б) Если в процессе распараллеливания программа была существенно оптимизирована/изменена/переписана (например, на С++) необходимо провести сравнение - исходная программа VS программа после преобразований.
в) Исследовать влияние различных опций оптимизации, которые поддерживаются компиляторами (-O2, -O3, -fast...)

6) Исследовать масштабируемость полученной параллельной программы: построить графики зависимости времени выполнения параллельной программы от числа используемых ядер для различного объёма входных данных.
Оптимальным является построение трёхмерного графика: по одной из осей время работы программы, по другой - количество ядер и по третьей - объём входных данных.
Такой график необходимо построить для каждого из разработанных вариантов программы.
Каждый прогон программы с новыми параметрами рекомендуется выполнять несколько раз с последующим усреднением результата (для избавления от случайных выбросов).
Для замера времени рекомендуется использовать функцию omp_get_wtime, общее время работы должно определяться временем работы самой медленной нити/процесса.
Количество ядер/процессоров рекомендуется задавать в виде p=1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100, 120, 140, 160.

7) Определить основные причины недостаточной масштабируемости программы при максимальном числе используемых ядер/процессоров.

8) Подготовить отчет о выполнении задания, включающий: описание реализованного алгоритма, графики зависимости времени исполнения от числа ядер/процессоров для различного объёма входных данных, текст программы.
