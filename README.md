# SEO-оптимизация карточек товара на маркетплейсах

## Checkpoint_1

Для первого чекпоинта был подготовлен проект со своим backend и tg ботом (@SeoProductCardsBot). Была обучена модель effecientnet-small для классификации футболки и джинсов (можно посмотреть в папке notebooks), в боте нужно будет написать команду `/check`. При запуске бота (команда `/start`) информация по пользователю сохранится в БД sqlite. Предсказание можно протестировать на своих фотографиях либо взять из папки test. Проект будет усложняться в процессе.

### Команды для запуска

Открыть два терминала для поднятие контейнеров backend и tg бота. 

Для backend:
1. `cd backend` - перейти в папку в backend
2. `poetry install --no-root --no-cache --only dev` - установить dev зависимости
3. `poetry run dvc pull` - подтянуть данные для модели
4. `docker build -t backend .` - собрать образ (хочу предупредить билдилось долго 1179.4s)
5. `docker run --rm -it --name backend -p 8000:8000 backend` - запустить контейнер

Для tg бота:

1. `cd telegram` - перейти в папку в tg бота
2. `docker build -t telegram . --build-arg="tg_token=ADD_YOUR_KEY"` - собрать образ (временно в гит вставил ключ, в конце проекта уберу)
3. `docker run --rm -it --name telegram telegram` - запустить контейнер

Чтобы получить доступ из одного докер-контейнера в другой докер-контейнер нужно добавить их в одну сеть (подсмотрел в этой [статье](https://habr.com/ru/articles/554190/)). 

1. `docker network create seo-network` - создаем сеть 
2. `docker network connect seo-network backend` - добавляем в сеть backend
3. `docker network connect seo-network telegram` - добавляем в сеть telegramm

И если теперь запустить инспектирование сети (`docker network inspect seo-network`), то в секции Containers мы увидим наши контейнеры.

Это все :), теперь ботом можно пользоваться!

## Checkpoint_2

Для второго чекпоинта была заиспользована БД PostgreSQL и настроено окружение в виде docker-compose файла.

Также я подключил pgAdmin, состояние БД можно посмотреть и через UI http://localhost:5050/ 

Параметры для подключения: 
- host `postgres_container`
- port `5432`
- dbname `seo-product-cards`
- user `kulikov`
- password `kulikov`

Сохранил для себя полезные статьи по настройке docker-compose: [тык1](https://habr.com/ru/articles/578744/), [тык2](https://habr.com/ru/articles/735274/).

- `docker-compose --project-name="seo-product-cards-pg-16" up -d` - команда по запуску docker-compose

- `docker-compose --project-name="seo-product-cards-pg-16" down` - команда по остановке docker-compose

Но посмотреть пункт выше: перед началом необходимо подтянуть данные модели!

# Студенты, выполняющие работы:
Бузаева Софья Михайловна, telegram: @ethee_real 

Куликов Дмитрий Алексевич, telegram: @dmitry_kulikov17

# Научный руководитель
Хажгериев Мурат Анзорович, telegram: @greedisneutral

# Постановка задачи
Идея: Помочь продавцам на маркетплейсах в заполнении seo карточки товара. Продавец делает фотографию товара и загружает его в сервис (телеграмм бот/web-приложение). Пользователю будет сгенерировано описание товара (можно, заранее задать ключевые слова) и подобрана наиболее подходящая категория. Правильно подобранная категория сильно повышает продажи на маркетплейсах. 

В будущем проект может быть использован для полноценного создания карточки товара на маркетплейсах с использованием их api методов. Делая только фотографию товара, пользователь может в несколько кликов выложить его на продажу сразу в нескольких торговых площадок, при этом не тратя время на изучение и особенностей их политик.

Допущения по ВКР: 
1. Рассматриваем два наиболее популярных маркетплейса в РФ: Ozon и Wildberries.
2. В рамках выполнения ВКР выберем только несколько видов категорий товаров, так как на площадках их присутсвует более 1000 штук (при этом список категорий динамический, со временем они могут добавляться/изменяться/удаляться). Это допущение делаем из за ограниченности во времени и вычислительных ресурсах.

# Этап работы:
1. Провести анализ данных. Написать парсеры маркетплейсов, чтобы собрать данные по товарам, категориям, описанию, фотографии и т.д. (20.03)
2. Подготовить проект по инфраструктурной части (настройка окружения, менеджемент зависимостей). (31.03)
3. Получаем из картинок эмбеддинги и обучаем на них классификатор с таргетом в виде категории товара. (31.04)
4. Также по этому эмбеддингу мы хотим получить описание товара (здесь можно воспользоваться готовыми АПИ или поднять свой инстанс LLM, которая будет генерить описание). (31.05)
5. Написание сервиса в виде телеграмм бота или web/приложения. (31.05)

# Анализ данных

Согласно [исследованию](https://www.retail.ru/news/tinkoff-ecommerce-v-2023-godu-kolichestvo-pokupok-na-marketpleysakh-vyroslo-na-6-29-yanvarya-2024-237087/) команды Tinkoff Ecommerce: самой привлекательной платформой для старта бизнеса является Wildberries: 63% продавцов в конце 2023 года выбирали ее в качестве первой площадки.

<img width="862" alt="image" src="https://github.com/Kulikov17/seo-optimization-product-cards/assets/61663158/66286c42-b768-4d40-a4ec-ae67740334e8">


Анализ парсинга Ozon:
1. Сильная защита, частая блокировка пользователей, смена userAgent не всегда помогает.
2. На странице много динамического контента и ленивой подгрузки, что создает трудности при парсинге динамически подгружаемых категорий.
3. Огромное количество подкатегорий товара.
4. Сложный парсинг товара:
   - Динамическая подгрузка отзывов, чтобы брать картинки из них.
   - Большая вариативность описания товара, нет единного шаблонна для парсинга. Так например, когда-то вместо текстов могут быть просто картинки или описание товара сопряженно с картинкой.


Анализ парсинга Wildberries:
1. Слабая защита, не требуется смена userAgent.
2. Статический контент для парсинга категорий и подкатегорий.
3. При парсинге товаров существует их ленивая подгрузка: для одной подкатегорий подгружается 15 товаров, чтобы подгрузить следующую пачку нужно проскроллить вниз.
4. Единное оформление карточки товара, упрощает парсинг товара: его описание, характеристики, фотографии из сео и из отзывов.


Итог: 
Принято решения писать ВКР для селлеров, использующий Wildberries, так как им чаще всего пользуются и он легче подается парсингу. В качестве направления дальнейшего развития можно рассмотреть другие маркетплейсы.


# Требования к запуску продукта:
  - Python 3.10
  - Poetry >= 1.0.0

## Стек технологий
Бэкенд будет написан на Python с использованием FastAPI. Модели будут написаны на pytorch. Для хранения фото и обработки фото может понадобится БД.

Если UI будет реализован в виде телеграмма бота, то предлагается использование библиотеки [aiogram](https://docs.aiogram.dev/en/latest/api/types/chat.html). В отличие от chatbot она предлагает использование ассинхронности, которая будет необходима при обработке и передачи фотографий и изображений.

Если UI будет реализован в виде web, то предлагается использование фреймворка Angular, так как есть опыт работы на нем, плюс позволяет создавать гибкие и масштабируемые решения.

<img width="627" alt="image" src="https://github.com/Kulikov17/seo-optimization-product-cards/assets/61663158/6bed9a33-21db-4950-8929-0916d7e52144">

## Запуск:
  1. `poetry install` установка зависимостей
  2. `poetry run pre-commit install`
  3. `poetry run dvc pull` получить все наборы данных и модели
  4. `poetry run train.py` запуск для обучения модели
  5. `poetry run infer.py` запуск для инференса
