from flask import Blueprint, render_template, current_app

web_bp = Blueprint("web", __name__)

@web_bp.route("/", methods=["GET"])
def home():
    """
    Renders frontend search interface template (index.html).
    Passes unique participant senders list to populate UI dropdown.
    """
    search_service = current_app.config["SEARCH_SERVICE"]
    senders = search_service.corpus_loader.get_unique_senders()
    return render_template("index.html", senders=senders)
