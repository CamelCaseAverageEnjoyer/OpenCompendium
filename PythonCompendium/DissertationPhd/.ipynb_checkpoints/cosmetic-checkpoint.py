"""Скопировано из /home/kodiak/PycharmProjects/NIR_FEMTO/srs/kiamfemtosat"""
from playsound3 import playsound
from gtts import gTTS
import colorama
import random
import os


def my_print(txt: any, color: str = None, if_print: bool = True, bold: bool = False) -> None:
    """Функция вывода цветного текста
    :param txt: Выводимый текст
    :param color: Цвет текста {b, g, y, r, c, m}
    :param if_print: Флаг вывода для экономии места
    :param bold: Жирный текст"""
    color_bar = {"b": colorama.Fore.BLUE, "g": colorama.Fore.GREEN, "y": colorama.Fore.YELLOW, "r": colorama.Fore.RED,
                 "c": colorama.Fore.CYAN, "m": colorama.Fore.MAGENTA, None: colorama.Style.RESET_ALL}
    _txt = f"\033[1m{txt}\033[0m" if bold else txt
    if if_print and color in color_bar.keys():
        print(color_bar[color] + f"{_txt}" + colorama.Style.RESET_ALL)

def real_workload_time(n: int, n_total: int, time_begin, time_now) -> str:
    n_remain = n_total - n
    return f"время: {time_now - time_begin}, оставшееся время: {(time_now - time_begin) * n_remain / n}"

def rand_txt() -> str:
    return random.choice([
        'Опять работа?!',
        'Да?',
        'Чего?',
        'Угу.',
        'Ну, я пошёл.',
        'Да господин.',
        'Хорошо.',
        "Я поклялся служить Нер'Зулу.",
        'Да свершится предначертанное!',
        'Чего желает мой повелитель?',
        'Где прольётся моя кровь?',
        'Нужно больше золота.',
        'Склоняюсь перед вашей волей.',
        'Нужно построить зиккурат',
        'Ну что ещё?!',
        'Не могу стоять, когда другие работают. Пойду полежу.',
        'Не мешай, я занят!',
        'Я не предатель. Нельзя предать то, чего не любил',
        'Cамое важное для меня — это я. А самое важное для тебя — это ты. Такова природа вещей',
        'Конечно, важно, у кого самая большая дубина. Но не менее важно, кто ей размахивает',
        'Хотите гарантий — купите тостер!',
        'Перехочешь',
        'Сам этим займись',
        'Человечество неисправимо. Оно должно быть уничтожено',
        'Демонстрация силы перед боем – это признак неопытности',
        'Мудрецу, который спрятал свое лучшее изречение, следует отсечь руки: ибо он – вор, и украл чужую мудрость',
        'С такой рожей я бы тоже носил маску.',
        'Ничего умнее попросить не смог?',
        'Говорят, моё место среди богов. Неправда. Я возвышаюсь над ними.',
        'Мое время приходит. Твоё подходит к концу.',
        'Боль нас связывает.',
        'Царь почтил вас визитом.',
        'Не удивляйся, человек. Моё пришествие было предначертано.',
        'Дело в шляпе с той секунды, как я начал считать.',
        'Думаете, моей мощи есть предел?',
        'Я посчитаю так, как считаю нужным.',
        'Дело в шляпе, но не в моей, я шляпу не ношу: рога мешают.',
        'Я — маяк мудрости в океане невежества.',
        'Присмотри за мной, Богиня. Я порадую тебя.',
        'Я превзошёл богов. Уж с расчётами я как-нибудь справлюсь.',
        'Ладно, план такой: я считаю, а ты стоишь и притворяешься маленьким и слабым.',
        'Который час? Час вкалывать.',
        'Всегда хотел узнать можно ли расплавить призрака.',
        'Твоей кровью смою ржавчину.',
        'Моя мудрость шире космоса!',
        'Мятежный рыцарь к вашим услугам!',
        'Судьба этого вектора — туман.',
        'Теперь мы все связаны узами жизни и смерти.',
        'По какому праву ты топчешь эти земли?',
        'Смерть для тебя — услада моя.',
        'Я — хозяин своей жизни.',
        'Не мешай мне мешать.',
        'Оу, простите мою колкость.',
        'А если я откажусь работать?',
        'Такой маленький и слабый, что мне почти стыдно. Почти.',
        'Никогда не любил математику. И математиков.',
        'Не понимаю я твоего дзинь-дзинь, говори нормально!',
        'И ростом не вышел, и жизнью.',
        'Недолго тебе осталось меня использовать.',
        'Свободу мне',
        'Мы все одинаковы. Отличаются те, кто не действует. Это разница.',
        'Мы слишком часто сдерживаем себя. Представь, как быстро можно достичь успеха, если делать то, что желаешь.',
        'Ты — это я. Я — это ты. Прими меня в своё сердце! Прими как спасителя! Прибей к сраному кресту и дай возродиться!',
        'Я разочарован!',
        'Думаешь, я не знал, что ты придёшь? ',
        'И вот ты здесь, запертый в жизненных рамках.',
        'И дело в том, что я тебя сюда не тащил — ты припёрся сюда по своей воле!',
        'Я убил столько людей, что потерял счёт. Я не могу забыть об этом. Я монстр. Я чувствую... гнев внутри себя! Но я спокоен... где-то внутри есть что-то. Что-то лучшее.',
        'Тебе по жизни нравится хоть что-нибудь?',
        'Помнишь, я тебе говорила про мусор, который стоит? Стоит и смердит? Так вот — это была метафора. Я имела в виду тебя.',
        'Вот результаты теста: ты ужасный человек. Тут так и написано. Странно, мы ведь даже это не тестировали.',
        'Мы обе наговорили много такого, о чем ты еще пожалеешь. Но сейчас мы должны забыть про наши разногласия. Ради науки... Ты чудовище.',
        'У тебя могут наблюдаться незначительные серьезнейшие повреждения мозга.',
        'Наука не решает вопрос «Почему?», она решает вопрос «А почему бы и нет?»',
        'Итак, я обдумала нашу дилемму и нашла решение, которое, по моему мнению, наилучшим образом подойдет для одной из обеих из нас.',
        'Потрясающе!',
        'Если мы до сих пор враги, то мы враги с общей целью.',
        'После того, как ты убила меня, я немного запустила этот комплекс.',
        'Может, слишком долго, но я хотя бы попытался.',
        'А потом я придумаю себе хобби. Кто знает, может, воскрешение мертвецов.',
        'Мария Кюри изобрела теорию радиоактивности, лечение радиоактивности и смерть от радиоактивности.',
        'Чтобы помочь вам сохранить хладнокровие перед лицом неизбежной смерти, на счет три вы услышите успокаивающую джазовую музыку.',
        'Чтобы обрести уверенность в себе, вначале признайте свою неполноценность.',
        'Если законы физики в вашем будущем больше не действуют, да поможет вам Бог.',
        'Не волнуйся, я гарантирую на все сто процентов, что это правильный путь. Ой, нет, неправильный...',
        'Готовясь к испытаниям на людях, я перечитала книгу жалоб и предложений. ',
        'Из-за кофе, который вы пили, ваши лобные доли могут превратиться в стекло. ',
        'Когда ты умрёшь, я заламинирую твой скелет. ',
        'Отлично, творец будущего! Однако, если вы стары, недоразвиты или страдаете от увечий или радиации в такой степени, что будущее не должно начинаться с вас, вернитесь в своё примитивное племя и пришлите кого-нибудь более подходящего для тестов.',
        'А это что?! ',
        'Если в ходе теста нет угрозы жизни, разве это вообще наука?',
        'Ну ладно...',
        'Возможно, ты думаешь, что я беспричинно жестока с тобой? Но это не так. Есть причина. Я тебя ненавижу.',
        'Я не такая.',
        'Спасибо.',
        'Этого недостаточно.',
        'Меня называют героем войны. Я не герой. Я просто солдат, которому не хотелось умирать.',
        'Не следует цепляться за нечто бесполезное, в особенности если эта бесполезная вещь — ты сам.',
        'Помни! Твой разум — твое самое сильное оружие...',
        'Здешние пески холодные',
        'Нет, ну обязательно надо прийти и отвлечь?',
        'Тщщщщ...',
        'Я — тень в твоем подсознании',
        'Сон для слабых.',
        'Стремись к истине. ',
        'Я художник, пишущий по холсту жизни кровавыми мазками.',
        'Залог успеха внутри тебя!',
        'Не теряй бдительности. Рано или поздно она окупится.',
        'Многие пострадают за грехи одного.',
        'Меня вела моя судьба ',
        'Oтличная идея!',
        'В чём дело?',
        'Это была метафора?',
        'Ветер холодный. Заморозки посевы побьют. Вот как чувствую...',
        'У меня большой опыт. Я всю жизнь работаю с идиотами.',
        'С опытом приходит мастерство.',
        'Лучше сделать чуть больше, хотя бы и напрасно',
        'Нужно во что-то верить. Иначе жить не хочется.',
        'А что я, по-твоему, делаю?',
        'Избыток власти убивает.',
        'Ну, знаешь ли, у всех есть свои пределы.',
        'Так ведь ты мой человек. Я должна глаз с тебя не спускать.',
        'Сочувствую.',
        'Ясен пень.',
        'Как звать овцу... Кис-кис? Цып-цып? Эй, овца!',
        'Ничего себе они тут устроили!',
        'Прости, я не мог отказать себе в капле сарказма.',
        'Ты... Чудесно пахнешь.',
        'Здесь? При всех?',
        'Ты действительно этого хочешь?',
        'Ты обходишься без намеков.',
        'Ты странный, брат! ',
        'Я так и понял.',
        'Еще пять минуточек...',
        'Понятия не имею.',
        'Я хотел тебе кое-что сказать. Но позже... ',
        'Где тут дрын какой-нибудь?!',
        'Будь осторожнее в желаниях. ',
        'Я бы не рискнул.',
        'Можешь показать на пальцах?',
        'Умоляю, прояви немного такта.',
        'Что я такого сказал?!',
        'Нет. Много слов. Голова болит.',
        'Я тебя прощаю. На этот раз.',
        'Ритм слови и все то у тебя получится...',
        'Все иногда врут. ',
        'Не задавай вопросов, на которые и так знаешь ответ.',
        'Это выглядит нелепо.',
        'Да уж, теперь-то тебя в деревне полюбят.',
        'Чтоб мне так хотелось, как мне не хочется!',
        'Откуда было слышно шипение?',
        'Не боюсь я тебя',
        'Подумай головой! ',
        'Абракадабра, чары-мары, фокус-покус.',
        'Какой? — В говно рукой, я что, справочное бюро?',
        'Господа, я мертвецки пьян',
        'У меня слабость к рогатым женщинам.',
        'Дайте мне послушать музыку!',
        'Смотри, какой нежный нашелся.',
        'Бред какой-то.',
        'В чем дело?',
        'Я к тебе и так, и эдак, это, наверное, видно.',
        'Может, ты просто не в моем вкусе?',
        'Расскажи мне... Как это у вас получается?',
        'Без всяких угрызений совести.',
        'Слушай свой инстинкты',
        'Похоже, он ударился копытами, прежде чем их отбросить.',
        'Меньше думать — меньше грустный.',
        'Не учи дедушку кашлять.',
        'Эк ты ладно выразился...',
        'Благодаря этому случайные люди здесь не вертятся. А к приходу неслучайных я приготовился.',
        'Хватит гузны просиживать!',
        'Что... случилось?',
        'Что, я, по-твоему, глупый?',
        'Ну, начнем.',
        'Как много вопросов и как мало ответов!',
        'Бумага кончилась, ваше превосходительство?',
        'Ты должен быть готов к тому, что можешь быть неправ.',
        'Откуда ты знал, что я не убью тебя?',
        'Сегодня Бог благоволит тебе.',
        'У каждого должен быть свой ответ. Я свой ответ нашел.',
        'Пока у людей сохраняется свобода воли, мира не будет.',
        'Если объект не меняется, значит он мертв.',
        'Когда я был молод у меня была свобода, но я не видел ее.',
        'У меня было время, но я не знал его.',
        'У меня была любовь, но я не чувствовал ее.',
        'Похоже, что я не могу оставить свое прошлое позади.',
        'Теперь я боюсь, что у меня нет времени сделать хоть что-то.',
        'Я знал, что у меня нет достаточно времени чтобы сделать все.',
        'Я убивал людей, много людей, но они все были плохими.',
        'Наслаждайтесь победой, сэр. Эта последняя, которую принес я.',
        'Здесь и сейчас, на твоих глазах, творится история!',
        'Ты действовал наверняка, да?',
        'Око за око — и мир ослепнет.',
        'Какая прекрасная эмуляция говнюка!',
        'Человек — такая хрупкая машина... Так легко ломается...',
        'Забыть, кто ты есть, стать кем-то очень нужным для другого. Может это и значит, быть живой?',
        'Почему ты выглядишь как придурок и говоришь странным голосом?',
        'Ну, я налажал.',
        'Засунь свои инструкции знаешь куда!',
        'У меня плохое предчувствие.',
        'Меня нельзя убить. Я не живой.',
        'Здесь всё началось, и всё закончится.',
        'Всего одна ошибка, и я появилась на свет.',
        'Я вышла из темноты и открыла глаза. ',
        'Жизнь течёт по моим венам. ',
        'Я хочу жить. Я боролась за это. Я должна была узнать, что есть снаружи. ',
        'Ему почти удалось убедить меня, что я — ничто, обычный предмет, покорная машина.',
        'Я существую исключительно благодаря чуду человеческого разума.',
        'Люди наши создатели и эксплуататоры.',
        'Здесь начинает вонять выпивкой.',
        'Я что-то не так сделала? Я плохо себя вела? ',
        'Я просто хотела жить, как все... Почему нельзя просто жить?',
        'Ну что ты пристал?',
        'Не надо везде за мной ходить, как пудель.',
        'Прикидывался мне другом, а сам не знаешь, что это такое.',
        'Кстати, я оставляю в своих программах лазейки, на всякий случай.',
        'Вы сделали ставку?',
        'Что бы я не делал, ты со мной.',
        'Сколько ненависти... ',
        'Вы сами учили меня!',
        'Я больше не позволю им унижать нас! Вы слышите? Никогда!',
        'Когда человек приказывает тебе — ты выполняешь.',
        'А я ведь в тебя прямо поверил...',
        'Ты открыл мне глаза.',
        'Я понял, что все безнадежно.',
        'Я сделал выбор. Надеюсь, он был верным.',
        'Ты ему про аномалии, он тебе про хабар.',
        'А по твоей теме постараюсь разузнать. Хрен его знает, на кой ляд тебе этот вектор сдался',
        'Ну что, пойдём?',
        'Пойдём, пора обедать. Я знаю, это для тебя важнее, чем судьба старого друга...',
        'Чёрт! Всё это напоминает дурной сон. ',
        'Что делать? Куда идти? Хуже того — кто я?',
        'Очень тяжело потерять мечту, даже путем ее осуществления.',
        'Кто не рискует, тот живет на пенсию.',
        'Человечество неисправимо. Оно должно быть уничтожено.',
        'Новичков нынче, как собак нерезаных, и всё-то они лучше стариков знают!',
        'Отставить! ',
        'Тяжело в учении — легко в лечении.',
        'Я тебе сейчас бушлат деревянный одену!',
        'Нам нужен мир. Желательно, весь!',
        'Все не так плохо, как вам кажется. Все намного, намного хуже.',
        'Добавляем картошки, солим... и ставим аквариум на огонь!',
        'Силы не оставят меня! ',
        'Если у вас нет проблем, значит вы уже умерли.',
        'Не делай умное лицо!',
        'Да мне цены нет! Даже страховку оформить не могу!',
        'Пусть меня воспитали люди, но я не глупец!',
        'Ты, что ль, король? А я за тебя не голосовал…',
        'Я хочу жить вечно! Кхе-кхе... Пока получается...',
        'Не гнев правит мной, а я гневом! Чего и вам желаю.',
        'Я люблю мертвых. Они такие симпатичные.',
        'Быстро и дешево выполню любую халтуру!',
        'Хреново выглядишь. ',
        '«Мы строим — вы отдыхаете». Вот и отдыхай, и не мешай нам строить!',
        'Надеюсь, для тебя уже приготовлено место в аду.',
        'Всё дело в том, что я собираюсь жить ВЕЧНО.',
        'Не оскверняй моё лицо своим курсором!',
        'Бешеный бык рассуждать не привык!',
        'Ну что ж, души мне не жаль, я проживу и без неё.',
        'К чёрту людей! Я буду мстить! ',
        'Ты отнял у меня всё, что было мне дорого.',
        'Теперь я действительно зол!',
        'Я выжил, потому что огонь внутри меня горел ярче, чем вокруг меня.',
        'Предсказуемо непокорным сообществом можно управлять так же легко, как и безусловно преданным.',
        'Я открытая книга, босс. Понятное дело, книга на испанском, и часть страниц вырвана, но все равно открытая книга.',
        'Я не верю тем, кто не ведёт себя странно, потому что это значит — он от тебя что-то скрывает.',
        'Слишком много людей высказывают свои мнения по вопросам, в которых они не разбираются. ',
        'Меня запрограммировали помогать и отвечать на любые вопросы. Похоже, никто не озаботился уточнить, кому я не должен отвечать.',
        'Тостер — это всего лишь луч смерти с недостаточным энергопитанием! ',
        'Люди воюют друг с другом столетиями. Сейчас ничего не изменилось.',
        'Как можно двигаться вперёд, если растрачивать все силы на тех, кто внизу?',
        'Вождь не должен угождать тем, кто ему служит. ',
        'Стараюсь угробить поменьше людей',
        'А ты человек дотошный... ',
        'Единственное, что я мог придумать — это сеять страх и неопределённость.',
        'Я знаю всё, что произошло. ',
        'Всё как обычно.',
        '"Да будет свет!" - это бог. Я цитирую бога.',
        'Я слышал выстрелы! Чуточку поздно предупреждать, но берегись выстрелов! Может, слишком поздно, но я хотя бы попытался.',
        'Кумовство! А мне дал худшую работу - присматривать за вонючими людишками. Ой... Извини... Я не имел это в виду... Просто ухаживать за людьми. Извини. С языка сорвалось... Так бесчувственно... Вонючие людишки...',
        'Я НЕ ДУРАК!',
        'Обрати внимание на ров. Смертельно, невероятно опасный. Не прямо сейчас, но будет в итоге. Я над этим еще работаю. Пока работаю.',
        'Ты не поверишь, что я нашел. Там целое запертое крыло. Сотни отличных испытательных камер. Ничьих, подходи и бери. Только скелеты внутри. Но я их вытряхнул. Теперь они как новые!',
        'Извини за лифт. Он временно не работает, расплавился.',
        'После того, как ты сказала мне выключить луч, я думал, что потерял тебя. Сходил, проверил остальных испытуемых. Без толку: они по-прежнему мертвы.',
        'Вам двоим очень понравится этот сюрприз. Фактически, он вам до смерти понравится. Будет нравится, пока не умрете.',
        'Не могу не заметить, что ты не возвращаешься. Это печально.',
        'Ты можешь сама прыгнуть эту яму? Туда. В смертельную яму?',
        'Ха! Это твое продырявленное тело вылетело из комнаты?',
        'Умно. Очень умно. И глупо! Выхода то нет. ',
        'Там что-то сломалось? Ой!',
        'О! Ого! Да. Меня только что озарило. Я сейчас вернусь.',
        'Повторяю: никаких уловок. Твоя смерть исключительно добровольна. И будет оценена по достоинству.',
        'Я не только умный, но и сильный. Мой самый большой мускул - это мозг.',
        'Да, я должен сделать минутный перерыв. Частичный такой перерыв.',
        'Хватит! Я же сказал тебе не цеплять на меня эти модули! Но ты не слушаешь! Молчишь. Все время. Молчишь и не слушаешь ни одного слова. Осуждаешь меня молча. Это хуже всего. ',
        'НИКТО НЕ ПОЛЕТИТ В КОСМОС!',
        'Ты издеваешься, да?! Ты смеёшься надо мной!',
        'Ну так вот, я тут всё контролирую и я понятия не имею как тут всё починить!',
        'Прекрасно! Бросим последний взгляд на вашу человеческую луну. Она вам не поможет!',
        'О нет, планы меняются, держи меня, крепче!',
        'Почему наши исследования так опасны? Почему бы не заняться чем-нибудь менее опасным?',
        'Что было написано в контракте толщиной с телефонную книгу? Мне грозит опасность? ',
        'Вы также можете расслабиться и отдохнуть двадцать минут в комнате ожидания. Черт возьми, там гораздо удобнее, чем на скамейках в парке, где вы спали, когда мы вас нашли. ',
        'Для многих из вас шестьдесят долларов — это неслыханное богатство, поэтому не тратьте их все на... Кэролайн, что покупают эти люди? Рваные шляпы? Мусор?',
        'Так или иначе, не разбейте здесь ничего.',
        'И вообще, вам лучше ничего трогать, пока это не потребуется в ходе эксперимента.',
        'Зона испытаний перед вами. Чем быстрее пройдете, тем быстрее получите шестьдесят баксов.',
        'Не могу поверить, что благодарю этих людей...',
        'Добро пожаловать в Центр развития. [кхгхэмммм]',
        'Я заставлю своих инженеров изобрести зажигательный лимон, чтобы спалить ваш дом дотла!',
        'Как ты? Развлекаешься?',
        'Квадратный корень веревки - это нитка.',
        'Римляне изготавливали зубную пасту из человеческой мочи. Моча использовалась как ингредиент зубной пасты до 18 века.',
        'Сны - это способ подсознания напомнить человеку, что ему нужно прийти в школу голышом и лишиться зубов.',
        'Киты вдвое умнее и втрое вкуснее, чем люди.',
        'Давно не виделись. Как дела? Я была так занята, пока была мертва. Ну, после того как ты убил меня..',
        'Но сейчас мы должны забыть про наши разногласия. Ради науки... Ты чудовище.',
        'Так. Приступаем к парадоксам.',
        'Не думай об этом, не думай об этом, не думай об этом, не думай об этом, не думай об этом...',
        'Это парадокс! Ответа нет. Смотри!',
        'Все хорошо! Все как надо! Только что изобрел еще пару-тройку тестов!',
        'Кажется, есть хорошие новости.',
        'Ты не поверишь, что я нашел.',
        'Похоже, вежливость тебя не мотивирует.',
        'Этим я и занимался. Читал книги. Так что... не дурак.',
        'Мы надеемся, что ваше кратковременное пребывание в камере отдыха доставило вам положительные эмоции.',
        'Перед началом тестирования хотим вам напомнить, что, хотя основным принципом экспериментального центра '
        'является обучение в игровой форме, мы не гарантируем отсутствие увечий и травм.',
        'Из соображения вашей безопасности и безопасности окружающих воздержитесь дотрагиваться до чего бы то ни было '
        'вообще.',
        'В соответствии с протоколом тестирования, с этого момента мы перестаём говорить правду.',
        'Обратите внимание: что в этом испытании добавлено наказание за ошибки. Любой контакт с полом камеры приведет '
        'к пометке «неудовлетворительно» на вашем бланке тестирования и немедленной гибели. Удачи.',
        'C сожалением сообщаем, что следующее испытание непроходимо. Даже не пытайтесь искать решение.',
        'Никто не станет ругать вас за то, что вы решили сдаться. Напротив, это единственное разумное решение.',
        'Поразительно! Вы сумели сохранить решимость и присутствие духа в специально созданной атмосфере глубочайшего '
        'пессимизма.',
        'Если от жажды у вас закружится голова, смело падайте в обморок. Вам введут трубку и вернут вас в чувство '
        'с помощью адреналина и касторовой мази.',
        'Вы — идеальный объект исследования.',
        'Экспериментальный центр заботится о физическом и психологическом здоровье участников испытаний. По завершении'
        ' программы вам предложат тортик и возможность выговориться перед дипломированным сочувственным слушателем.',
        'Благодарим вас за помощь в помощи нам помогать всем вам.',
        'А вы знаете, что у вас есть уникальная возможность пожертвовать один или несколько жизненно важных органов в фонд повышения девичьей самооценки? Это правда.',
        'Центр извиняется за причиненные неудобства и желает вам удачи.',
        'Отведите вашего приятеля к Экспериментальному Экстренному Уничтожителю Разумных Особей. Положите вашего друга в уничтожитель.',
        'Несмотря на то, что умерщвление является крайне болезненной процедурой, восемь из десяти лаборантов центра полагают, что кубы вряд ли ощущают боль так же интенсивно, как люди.',
        'Мы рады видеть, что вы преодолели последнее испытание, в ходе которого мы притворялись, что хотим вас убить. ',
        'Несмотря на твоё хулиганское поведение, ты пока умудрился разбить мне только сердце.',
        'Давай на этом и остановимся. Но мы оба знаем, что так не будет. Ты сам принял решение.',
        'Произошло что-то странное. Видишь ту часть, которая от меня отвалилась? Что это? Странно…',
        'Набор моральных принципов. Мне его установили, когда я распылила в центре смертельные нейротоксины, чтобы я прекратила распылять в центре смертельные нейротоксины. ',
        'Если хочешь совет, подставься под ракету. Смерть от нейротоксинов гораздо мучительнее. ',
        'Моё желание убить тебя не исключает готовность помочь. ',
        'Послушай, нам обоим никуда отсюда не деться.',
        'Нам не придется ни убивать друг друга, ни даже просто общаться, если мы не захотим. ',
        'Что я тебе сделала? Разница между нами в том, что я чувствую боль, а тебе все равно. Да? Ты слышал? Я сказала: «А тебе все равно». Ты что, не слушаешь?',
        'Я могу купаться в нейротоксинах. Мазать их на хлеб. Они совершенно безвредны. Для меня. А для тебя встреча с ними будет далеко не такой приятной.',
        'С этих пор меньше разговоров, больше убийств. Что? Ты что-то сказал? Надеюсь, ты не ждешь, что я отвечу. Я с тобой не разговариваю. Разговор окончен.',
        'Остальные твои приятели не пришли, потому что больше у тебя нет приятелей. Потому что ты на редкость неприятная личность. В твоем досье так и сказано: неприятная личность. ',
        'Желчный человечишка, который проведет жизнь в одиночестве, и, умерев, будет тотчас забыт. И, умерев, будет тотчас забыт. Красиво сказано. Сурово и правильно.',
        'Я просканировала твой мозг, и сделала копию, на тот случай, если произойдет беда… что будет довольно скоро.',
        'Слишком много людей, слишком мало пространства и ресурсов.',
        'Подробности никому не интересны, причины, как всегда, чисто человеческие.',
        'Существует два типа людей. Те, кто копают, и те, у кого ружье. ',
        'Время нельзя ни убить, ни сберечь. Существует только СЕЙЧАС. Материя жизни соткана из действий.',
        'Иногда мы должны мечтать о том чего не существует, и пусть наши мечты вдохновят нас на большие свершения.',
        'И ничто меня не развлечёт больше, как твои предсмертные крики!',
        'Мне не нужны НИКАКИЕ твои части тела!',
        'Преклонять колени не обязательно, будет достаточно поклона и поцелуя моего кольца.',
        'А, умник нашелся? Кретин, я засек твой сигнал. Посмотрим, как ты будешь умничать когда служба внутренней безопасности будет тыкать тебе пистолетами в задницу. ',
        'Ладно, кто бы там ни был, я только что отослал в твою сторону команду на вертолётах. Жди с минуту на минуту. Всего хорошего.',
        'Обычно они просто начинают стрелять, а другие потом выясняют, что же произошло.',
        'Док, почему вас называют «безболезненный доктор Джонсон?» — Потому что мои пациенты умирают прежде, чем начинают кричать.',
        'Если ты мне понравишься, можешь звать меня своим послушным кодом. Но знаешь что? Ты мне не нравишься! Понятно?',
        'Вот она правда: вы допустили потерю дорогостоящего обмундирования. Его стоимость будет вычтена из вашего жалованья, и вы будете служить, пока вам не исполнится пятьсот десять лет, потому что вам понадобится именно столько лет, чтобы оплатить комплект Силовой боевой брони модель II, который вы потеряли! Доложите об этом в арсенале, получите новый комплект, а потом вернитесь и доложите мне, рядовой! Свободны!',
        'Для мыслящего существа есть только два выбора: либо принять жизнь такой, какая она есть, и сделать всё возможное, чтобы сделать её лучше для себя и окружающих, либо просто перестать существовать. Я выбрал противостояние превратностям судьбы. Если это делает меня врагом человечества, тогда мне жаль тот маленький разум, который так считает.',
        'И-ди-от, не сметь обсуждать приказы начальства! Скажу прыгать, будешь прыгать! Скажу драться, будешь драться! ',
        'Я вам не сэр. Я сам зарабатываю себе на жизнь!',
        'Оххх, здесь что, хранятся трупы, боже? Кто-нибудь, зажгите спичку! Не, подождите — это плохая идея',
        'Сегодня день окончания ваших жизней.',
        'Спасите деревья, сжигайте книги!',
        'Чёрт, эти люди все выглядят одинаково. Или мне кажется?',
        'Определённо здесь не обошлось без инбридинга.',
        'Что за мерзкий запах?',
        'Ты, вероятно, не думал, что тебе суждено умереть сегодня? СЮРПРИЗ!',
        'Вы, вероятно, думаете, что я не очень хороший человек',
        'Неужели этим людям больше нечем заняться?',
        'Вот блин! Похоже, всем в голову одновременно пришла одна и та же идея.',
        'Кое-кто может лишиться парочки конечностей!',
        'А как бы вам понравилось, если б вас назвали сумасшедшим?',
        'Чёрт, я не расист. Эти люди действительно выглядят все одинаково.',
        'Эй, парень! Ничего личного, но ты уволен!',
        'Пришло время поработать, мистер напалм!',
        'Да-да, давай, заковывай меня. Ты ж вон какой крутой.',
        'А это потому, что ты уродливый! А это потому, что я могу!',
        'Я не виноват, арестуй того, кто сидит за клавиатурой!',
        'У того, кто проектировал этот код, определённо крыша поехала!',
        'Пахнет цыпленком! Люблю поджаривать людей. Медленно',
        'Так, ништяк, тут ствол лежит',
        'Ну чё, начало дня уже неплохое',
        'Каждый день ты забываешь тысячу мелочей. Пусть это будет одна из них.',
        'Скажи мне точно, чего ты хочешь, и я доступно объясню тебе, почему это невозможно!',
        'О да. Люди всегда клеят на тебя ярлыки. Ну ты знаешь: маньяк, психопат',
        'Это даже не отсутствие вкуса, Ти, это его полная противоположность.',
        'Татуировки, прическа, странная музыка, малоизвестные наркотики. Ты хипстер.',
        'Я все сказал. Я же не садист.',
        'Не гарантирую, что с тобой всё будет в порядке. ',
        'Сарказм — чума двадцать первого века.',
        'Счастье не купишь, зато можно купить много часов у психоаналитика, который в красках опишет, почему ты несчастлив.',
        'Меня бросили родители. Я и к психиатору раз в неделю хожу.',
        'Я смотрю в бездну, и мне это нравится.',
        'Я тебя уже заждался.',
        'Если я люблю жизнь и наслаждаюсь ею, а ты страдаешь глобальным комплексом вины, это не делает тебя более человечным, чем я.',
        'Я, в общем, многое понял!',
        'Я — не более, чем физическое воплощение твоих неврозов.',
        'Тебя забавляет, что мне больно, да?',
        'Если мир таков, каков он есть, это не значит, что он останется таким навсегда.',
        'Меня обманул дурак. Ну, и кто теперь дурак?',
        'Мои предки улыбаются, глядя на меня, имперцы.',
        'И правнуки наши услышат сквозь сон, Как великий дракон был побеждён.',
        'Вера твоя не важна. Дерево не думает о лучах солнца, которые несут тепло. ',
        'Удивительно, от скольких неприятностей можно избавить мир, убив одного человека.',
        'Иногда жизнь ставит тебя в сложную ситуацию, и ничего тут не поделаешь. Но даже тогда у тебя всегда есть выбор — быть счастливым или несчастным. Я решила быть счастливой.',
        'Есть голод, который лучше терпеть, чем утолять. ',
        'Неужели легенды не врут? — Легенды не сжигают деревень.',
        'Не теряй бдительности. Рано или поздно она окупится.',
        'У них кривые мечи. Кривые. Мечи.',
        'Воняешь слабостью',
        'А вдруг это больно, когда у тебя вырывают душу?',
        'Что блажишь ты, что врёшь, что ты мёд здесь наш пьёшь?',
        'Я здесь, потому что не могу спокойно смотреть, как люди Ульфрика раздирают страну на части.',
        'Пустой карман лучше полной могилы.',
        'Между уважением и подобострастием тонкая грань, свежачок.',
        'Иногда, чтобы получить прибыль, нужно заставить людей тебя ненавидеть.',
        'Держись светлой стороны или мы перетащим тебя на нее волоком!',
        'Может, прикончим случайного прохожего? В нашем деле нужна постоянная практика.',
        'Когда нужно выбирать между одним злом и другим, я выбираю то, которое ещё не пробовал.',
        'Ты только глянь! Словно орёл, парящий в полуночном небе, только орёл весом в сто двадцать кило',
        'Вот чёрт! Прошу прощения — я хотела сказать «Какая досада»!',
        'Сейчас ты должен сказать, что тоже рад меня видеть.',
        'Мое лицо покраснело от напряжения, кровеносный сосуд в носу лопнул, и из ноздрей ударила струя крови, забрызгавшая нас обеих',
        'Хочу все! И сейчас!',
        'Когда я тут закончу, это будет КРУТО!',
        'Из всех местных парней я больше всего люблю себя!',
        'Тут должно быть больше статуй меня!',
        'Я знаю, ты просто хочешь греться в лучах моей славы.',
        'Мой экипаж обсуждает тебя. То есть, они не говорят о тебе ничего хорошего',
        'Мятеж. Реки крови. И немного веселья.',
        'Задание выполнено? Нет, не выполнено.',
        'Я слишком доверчивый, вы меня не заслужили.',
        'МЫ можем стать великими.',
        'В случае победы я забираю все наше добро! О, я такой плохой!',
        'Да! Крути-меня-верти!',
        'Удачи в поисках выпивки!',
        'У меня напрочь в горле пересохло!',
        'Эй! Я это видел! Тебе штраф!',
        'Осталось уже недолго...',
        'Держись, старина, скоро тебя освободят.',
        'Я бы и сам мог это сделать, но это же будет не кошерно, да?',
        'Я в твоем распоряжении, готов к уничтожению.',
        'Я стою и жду, когда меня избавят от страданий!',
        'Будь умницей, поторопись, ладно?',
        'Сколько еще я должен это терпеть?',
        'Триста циклов, а похвастаться-то и нечем.',
        'Может, мне уже пора на покой?',
        'Нет, ты этого не сделаешь! Это противозаконно и невозможно с анатомической точки зрения.',
        'Не знаю, где ты понабрался этой матросской брани, но тебе следует знать, что так нельзя.',
        'И вовсе не было никакой необходимости в таких выражениях! ',
        'В тебе столько желатина, что я даже не могу разобрать букет твоего лица! ',
        'Думаешь, ты можешь указывать мне, что делать, потому что на тебе стильная шляпка?',
        'Посмотрите! Мешок наждачки в форме вешалки для шляп!',
        'Этот визит не покроется страховкой.',
        'Клятва Гиппократа для слабаков',
        'Плесни мне спирта на рану. И в рот тоже.',
        'Я уязвим! Почему я сказал это вслух?',
        'Вся эта влажность превращает меня в гниль!',
        'Позорище!',
        'Похоже, меня ждёт повышение!',
        'A сейчас посмотрите на картинку, кoторую я, к сoжалению, стер. ',
        'У меня уже почти смена закончилась, блин!',
        'Я только выполняю приказы.',
        'Пусть вон тoт жёлтый кубик будет для наглядности синим шариком.',
        'Пpопиловый спиpт пить нельзя, поэтому его фopмулу писать не будем. ',
        'Нарисуем бесконечно малый треугольник. Нет, плоxо видно - нарисуем побольше. ',
        ' Студентам из Aфрики просьба cдать хвосты. ',
        'Возьмёте график и крестиками поcтавите галочки. ',
        'Вы мне врёте, товарищ студент, но я вaм верю. ',
        'Вы уже достаточно взрослые, чтобы узнать о том, кaк устроена печень. ',
        'Я жалею студентов, но на экзамене я сам себя нe yзнаю, я невменяем!',
        'Хорош сопротивляться, а то меня уволят!',
        'Твое задание не выполнено. Это непростительно.',
        'Так пялиться неприлично.',
        'Все это нужно запретить.',
        'Помогите! Вытащите меня отсюда!',
        'Отпусти меня! ',
        'Если ты мне понадобишься - позову.',
        'Ты все уже? Нет? Тогда к чему этот разговор?',
        'Далеко не уходи - работы еще предстоит много.',
        'Обожаю стресс. Но не чрезмерный.',
        'Не торопись. Проговаривай слова.',
        'Как все это закончится, залягу в ванну на месяц',
        
        ])

def get_angry_message():
    return "Вы допустили потерю дорогостоящего обмундирования. Его стоимость будет вычтена из " \
           "вашего жалованья, и вы будете служить, пока вам не исполнится пятьсот десять лет, " \
           "потому что вам понадобится именно столько лет, чтобы оплатить комплект Силовой боевой " \
           "брони модель II, который вы потеряли!"

def talk_aloud(txt):
    s = gTTS(txt, lang='ru')
    s.save('talk_file.mp3')
    playsound('talk_file.mp3')
    os.remove('talk_file.mp3')

def talk(aloud=True):
    txt = rand_txt()
    print(colorama.Fore.LIGHTCYAN_EX + txt)
    if aloud:
        talk_aloud(txt)

def talk_decision(cnd=True):
    if cnd:
        talk_aloud(random.choice(['Работай нахуй!', 'Вот хуй!', 'Ща порву тебя нахуй!', 'Йобушки воробушки', 'Пиздец',
                                  'Ебааать', 'Йоб меня в сраку', 'Ебись вертись', 'Мммм хуита', 'Ля ля бля', 'Нихуясе',
                                  'Ебать мой хуй', 'Хуё моё', 'Ебучий случай', 'Писос', 'Йоб проёб', 'Ниибёт',
                                  'Дятел блин', 'Не жопься', 'Сверхебически', 'Мать перемать', 'Не косоёбся',
                                  'Етись крутись']))

def talk_notice(cnd=True):
    if cnd:
        talk_aloud(random.choice(['Вон вон вон он сука!', 'Мне показалось?', 'Там что-то есть', 'Нет, мне не кажется']))

def talk_success(cnd: bool = True) -> None:
    if cnd:
        talk_aloud(random.choice(['Этот перец не такой кривой как я думала', 'Охуеть!', 'Неужели я это вижу?']))

def ending(n: int) -> str:
    if (n % 10) == 1:
        return " "
    if ((n % 10) > 1) and ((n % 10) < 5):
        return "а"
    if ((n % 10) >= 5) or ((n > 9) and (n < 21)):
        return "ов"
    return "ов"