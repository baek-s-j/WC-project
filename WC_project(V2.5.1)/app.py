from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
import threading, time
import winsound as sd
import math

app = Flask(__name__)
app.secret_key = '2017'


# def ex():
#     print("되냐?")
#     fr = 2000  # range : 37 ~ 32767
#     du = 500  # 1000 ms ==1second
#     sd.Beep(fr, du)  # winsound.Beep(frequency, duration)
#
#     threading.Timer(10, ex).start()
#
#
# ex()


@app.route('/')
@app.route('/index.html')
def index():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    client.close()
    return render_template('index.html', count=count_list)


@app.route('/visualization.html')
def visualization():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    z_result = list(collection.find())
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    client.close()
    z_chart = ["../static/chart/무위사-내부-후불벽화-5번.html", "../static/chart/무위사-외부-전면-3번.html",
               "../static/chart/무위사-외부-후면-4번.html", "../static/chart/무위사-내부-우측-2번.html",
               "../static/chart/무위사-내부-좌측-1번.html"]
    return render_template('visualization.html', count=count_list, z_result=z_result, z_chart=z_chart)


@app.route('/addManager.html')
def addManager():
    return render_template('addManager.html', )


@app.route('/addManagerAct', methods=['POST', 'GET'])
def addManagerAct():
    if request.method == 'POST':
        client = MongoClient('127.0.0.1', 27017)
        db = client.wc_project
        collection = db.manager
        result = request.form.to_dict()
        id_check = list(collection.find({'manager_id': result.get('manager_id')}))
        if not id_check:
            number = collection.estimated_document_count()
            result['manager_number'] = number + 1
            collection.insert_one(result)
            client.close()
            flash("회원가입이 완료되었습니다.")
            return redirect(url_for("index"))
        else:
            flash("아이디가 중복되었습니다.")
            return redirect(url_for("addManager"))


@app.route('/login.html')
def login():
    return render_template('login.html')


@app.route('/loginAct', methods=['POST', 'GET'])
def loginAct():
    if request.method == 'POST':
        client = MongoClient('127.0.0.1', 27017)
        db = client.wc_project
        collection = db.manager
        result = request.form.to_dict()
        print(result)
        info = list(collection.find({'manager_id': result.get('manager_id')}))
        print(info)
        client.close()
        if result.get('manager_password') == info[0]['manager_password']:
            session['id'] = result.get('manager_id')
            return redirect(url_for("index"))

        else:
            flash("아이디 또는 비밀번호가 맞지 않습니다.")
            return redirect(url_for("login"))


@app.route('/logout')
def logout():
    session.pop('id', None)
    return redirect(url_for("index"))


@app.route('/ML.html')
def ML():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    client.close()
    return render_template('ML.html', count=count_list)


@app.route('/404.html')
def err():
    return render_template('404.html')


@app.route('/blank.html')
def blank():
    return render_template('blank.html')


@app.route('/button.html')
def button():
    return render_template('button.html')


@app.route('/chart.html')
def chart():
    return render_template('chart.html')


@app.route('/element.html')
def element():
    return render_template('element.html')


@app.route('/form.html')
def form():
    return render_template('form.html')


@app.route('/signin.html')
def signin():
    return render_template('signin.html')


@app.route('/signup.html')
def signup():
    return render_template('signup.html')


@app.route('/table.html')
def table():
    return render_template('table.html')


@app.route('/typography.html')
def typography():
    return render_template('typography.html')


@app.route('/widget.html')
def widget():
    return render_template('widget.html')


@app.route('/addRelic.html')
def addRelic():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    client.close()
    return render_template('addRelic.html', count=count_list)


@app.route('/addRelicAct', methods=['POST', 'GET'])
def addRelicAct():
    if request.method == 'POST':
        client = MongoClient('127.0.0.1', 27017)
        db = client.wc_project
        collection = db.relic
        result = request.form.to_dict()
        number = collection.estimated_document_count()
        result['relic_number'] = number + 1
        collection.insert_one(result)
        client.close()
        flash("성공")
        return redirect(url_for("addRelic"))


@app.route('/addSensor.html')
def addSensor():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    client.close()
    return render_template('addSensor.html', count=count_list)


@app.route('/addSensorAct', methods=['POST', 'GET'])
def addSensorAct():
    if request.method == 'POST':
        client = MongoClient('127.0.0.1', 27017)
        db = client.wc_project
        collection = db.sensor
        result = request.form.to_dict()
        number = collection.estimated_document_count()
        result['sensor_number'] = number + 1
        collection.insert_one(result)
        client.close()
        flash("성공")
        return redirect(url_for("addSensor"))


@app.route('/addZone.html')
def addZone():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    client.close()
    return render_template('addZone.html', count=count_list)


@app.route('/addZoneAct', methods=['POST', 'GET'])
def addZoneAct():
    if request.method == 'POST':
        client = MongoClient('127.0.0.1', 27017)
        db = client.wc_project
        collection = db.zone
        result = request.form.to_dict()
        number = collection.estimated_document_count()
        result['zone_number'] = number + 1
        collection.insert_one(result)
        client.close()
        flash("성공")
        return redirect(url_for("addZone"))


@app.route('/datatable.html')
def datatable():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    page = request.args.get('page')
    tab = request.args.get('tab')
    if page is None:
        page = 1

    page = int(page)
    if tab == 'relic':
        collection = db.relic
        number3 = collection.estimated_document_count()
        r_result = list(collection.find().skip((page - 1) * 10).limit(10))

        collection = db.sensor
        number1 = collection.estimated_document_count()
        s_result = list(collection.find().limit(10))
        collection = db.zone
        number2 = collection.estimated_document_count()
        z_result = list(collection.find().limit(10))
        collection = db.manager
        number4 = collection.estimated_document_count()
        m_result = list(collection.find().limit(10))

    elif tab == 'zone':
        collection = db.relic
        number3 = collection.estimated_document_count()
        r_result = list(collection.find().limit(10))

        collection = db.sensor
        number1 = collection.estimated_document_count()
        s_result = list(collection.find().limit(10))
        collection = db.zone
        number2 = collection.estimated_document_count()
        z_result = list(collection.find().skip((page - 1) * 10).limit(10))
        collection = db.manager
        number4 = collection.estimated_document_count()
        m_result = list(collection.find().limit(10))

    elif tab == 'manager':
        collection = db.relic
        number3 = collection.estimated_document_count()
        r_result = list(collection.find().limit(10))

        collection = db.sensor
        number1 = collection.estimated_document_count()
        s_result = list(collection.find().limit(10))
        collection = db.zone
        number2 = collection.estimated_document_count()
        z_result = list(collection.find().limit(10))
        collection = db.manager
        number4 = collection.estimated_document_count()
        m_result = list(collection.find().skip((page - 1) * 10).limit(10))

    else:
        collection = db.relic
        number3 = collection.estimated_document_count()
        r_result = list(collection.find().limit(10))

        collection = db.sensor
        number1 = collection.estimated_document_count()
        s_result = list(collection.find().skip((page - 1) * 10).limit(10))
        print(s_result)
        collection = db.zone
        number2 = collection.estimated_document_count()
        z_result = list(collection.find().limit(10))
        collection = db.manager
        number4 = collection.estimated_document_count()
        m_result = list(collection.find().limit(10))

    count_list = list((number1, number2, number3, number4))
    page_count1 = number1 / 10
    page_count1 = math.ceil(page_count1) + 1
    page_count2 = number2 / 10
    page_count2 = math.ceil(page_count2) + 1
    page_count3 = number3 / 10
    page_count3 = math.ceil(page_count3) + 1
    page_count4 = number4 / 10
    page_count4 = math.ceil(page_count4) + 1
    page_list = list((page_count1, page_count2, page_count3, page_count4))
    print(page_list)
    client.close()
    return render_template('datatable.html', r_result=r_result, s_result=s_result, z_result=z_result, m_result=m_result,
                           count=count_list, page_list=page_list)


@app.route('/THDatatable.html')
def THDatatable():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    z_result = list(collection.find())
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    collection = db.thdata
    th_tmp = list()
    for i in z_result:
        tmp = list(collection.find({"zone_name": i['name']}))
        th_tmp.append(tmp)

    client.close()
    return render_template('THDatatable.html', count=count_list, z_result=z_result, th_result=th_tmp)


@app.route('/THDatatableAct.html', methods=['POST', 'GET'])
def THDatatableAct():
    if request.method == 'POST':
        result = request.form.to_dict()

        month = "-" + str(format(int(result['month']), '02'))
        day = "-" + str(format(int(result['day']), '02'))
        if month == "-00":
            month = ""
        if day == "-00":
            day = ""

        query = result['year'] + month + day
        print(query)
        client = MongoClient('127.0.0.1', 27017)
        db = client.wc_project
        collection = db.relic
        number3 = collection.estimated_document_count()
        collection = db.sensor
        number1 = collection.estimated_document_count()
        collection = db.zone
        z_result = list(collection.find())
        number2 = collection.estimated_document_count()
        collection = db.manager
        number4 = collection.estimated_document_count()
        count_list = list((number1, number2, number3, number4))
        collection = db.thdata
        th_tmp = list()
        for i in z_result:
            tmp = list(collection.find({"zone_name": i['name'], "time": {"$regex": query}}))
            th_tmp.append(tmp)

        client.close()
        return render_template('THDatatableAct.html', count=count_list, z_result=z_result, th_result=th_tmp,
                               day_re=result)


@app.route('/riskDatatable.html')
def riskDatatable():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    z_result = list(collection.find())
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    collection = db.riskdata
    risk_tmp = list()
    for i in z_result:
        tmp = list(collection.find())
        risk_tmp.append(tmp)

    client.close()
    return render_template('riskDatatable.html', count=count_list, z_result=z_result, risk_result=risk_tmp)


@app.route('/tlqkfwha.html')
def tlqkfwha():
    client = MongoClient('127.0.0.1', 27017)
    db = client.wc_project
    collection = db.relic
    number3 = collection.estimated_document_count()
    collection = db.sensor
    number1 = collection.estimated_document_count()
    collection = db.zone
    z_result = list(collection.find())
    number2 = collection.estimated_document_count()
    collection = db.manager
    number4 = collection.estimated_document_count()
    count_list = list((number1, number2, number3, number4))
    collection = db.ex_sigac
    th_tmp = list(collection.find().limit(1000))
    client.close()
    return render_template('tlqkfwha.html', count=count_list, z_result=z_result, th_result=th_tmp)


# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
