# SEO-оптимизация карточек товара на маркетплейсах

Идея: Помочь продавцам на маркетплейсах в заполнении seo карточки товара. Продавец делает фотографию товара и загружает его в сервис (телеграмм бот/web-приложение). Пользователю будет сгенерировано описание товара (можно, заранее задать ключевые слова) и подобранна наиболее подходящая категория. Правильно подобранная категория сильно повышает продажи на маркетплейсах. 

В будущем проект может быть использован для полноценного создания карточки товара на маркетплейсах с использованием их api методов. Делая только фотографию товара, пользователь может в несколько кликов выложить его на продажу сразу на нескольких маркетплейсах, при этом не тратя время на изучение и особенностей политик разных торговых площадок.

Допущения: 
1. Рассматриваем два наиболее популярных маркетплейса в РФ: Ozon и Wildberries.
2. В рамках выполнения ВКР выберем только несколько видов категорий товаров, так как на площадках их присутсвует более 1000 штук (при этом список категорий динамический, со временем категории могут добавляться/изменяться/удаляться). Это допущение делаем из за ограниченности времени и вычислительных ресурсов.

Этап работы:
1. Минимально подготовить проект по инфраструктурной части (настройка окружения, менеджемент зависимостей). Также подготовить скелет кода в виде решения задачи: классификация одежды с использованием датасета [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). Этот пункт больше нужен для выполнения ДЗ задания по курсу MLOPS, но является хорошей начальной точкой для старта реализации проекта.
2. Написать парсеры маркетплейсов, чтобы собрать данные по товарам, категориям, описанию, фотографии и т.д.
3. Получаем из картинок эмбеддинги и обучаем на них классификатор с таргетом в виде категории товара.
4. Также по этому эмбеддингу мы хотим получить описание товара (здесь можно воспользоваться готовыми АПИ или поднять свой инстанс LLM, которая будет генерить описание).
5. Написание сервиса в виде телеграмм бота или web/приложения.
