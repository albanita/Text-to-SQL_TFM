from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from scripts.Traductor_ES_EN import *
from scripts.Trained_Text_to_SQL_Model import *
from scripts.DB_Connection import *

load_dotenv()
app = Flask(__name__, static_folder="static", template_folder="templates")

global TEXT_TO_SQL_MODEL
global TRADUCTOR_MODEL


@app.route("/")
def index():
    return render_template('index.html', data=None)


@app.route('/generate_sql', methods=['POST'])
def generate_sql():
    data = request.json
    input_nl_query = data.get('nl_query', '')
    query_in_spanish = data.get('query_in_spanish', False)

    global TEXT_TO_SQL_MODEL
    global TRADUCTOR_MODEL

    if query_in_spanish:
        try:
            input_nl_query = TRADUCTOR_MODEL.translate(input_nl_query)
        except Exception as e:
            return jsonify({
                'sql_query': str(e),
                'columns_list': None,
                'values_list': None,
                'show_modal': False
            })

    sql_query = TEXT_TO_SQL_MODEL.generate_sql(input_nl_query=input_nl_query)

    try:
        db = DB_Connection()
        if "SELECT name SELECT name" in sql_query:
            sql_query = sql_query.split("SELECT name")[1]
        columns_list, values_list = db.execute_query(sql_query)

        return jsonify({
            'sql_query': sql_query,
            'columns_list': columns_list,
            'values_list': values_list,
            'show_modal': True
        })
    except Exception as e:
        return jsonify({
            'sql_query': str(e),
            'columns_list': None,
            'values_list': None,
            'show_modal': False
        })


if __name__ == "__main__":
    global TEXT_TO_SQL_MODEL
    global TRADUCTOR_MODEL

    TEXT_TO_SQL_MODEL = Trained_Text_to_SQL_Model(model_path="model_assets/text_to_sql_model.h5",
                                                  input_tokenizer_path="model_assets/input_tokenizer.json",
                                                  target_tokenizer_path="model_assets/target_tokenizer.json")

    # text = "Display the mission_name and mission_status for all where start_date is before 1990 in the space_missions table"
    # # text = "Display the mission_name, launch_date, and mission_status for all missions launched before 1990 in the space_missions table"
    # sql_query = TEXT_TO_SQL_MODEL.generate_sql(input_nl_query=text)
    # print(sql_query)

    TRADUCTOR_MODEL = Traductor_ES_EN()

    app.run(debug=True)
