from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import markdown2
from django.http import HttpResponse
import logging
from chatbot.romance_chatbot import process_romance_chatbot_request
from wishlist.utils import extract_titles, save_recommended_works
from chatbot.historical_chatbot import process_historical_chatbot_request

logging.basicConfig(level=logging.INFO)


def basic_chatbot_na_view(request):
    return render(request, "chatbot/basic_chatbot_na.html")


@login_required
def basic_chatbot_view(request):
    return render(request, "chatbot/basic_chatbot.html")


@login_required
def romance_chatbot_view(request):
    question = request.GET.get("question", "").strip()
    session_id = request.session.get("session_id")
    if not session_id:
        request.session.create()
        session_id = request.session.session_key
        request.session["session_id"] = session_id
    logging.info(f"session_id: {session_id}")

    if not question:
        profile_image_url = "/static/img/romance/default_user.png"
        return render(
            request,
            "chatbot/romance_chatbot.html",
            {"profile_image_url": profile_image_url},
        )

    response = process_romance_chatbot_request(question, session_id, request.user)
    for step in response:
        logging.info(step)
    final_output = step["output"]
    recommended_titles = extract_titles(final_output)
    if recommended_titles:
        logging.info(recommended_titles)
        save_recommended_works(request.user, recommended_titles, "romance")
    return HttpResponse(markdown2.markdown(final_output))


@login_required
def rofan_chatbot_view(request):
    return render(request, "chatbot/rofan_chatbot.html")


@login_required
def fantasy_chatbot_view(request):
    return render(request, "chatbot/fantasy_chatbot.html")


@login_required
def historical_chatbot_view(request):
    question = request.GET.get("question", "").strip()
    session_id = request.session.get("session_id")
    if not session_id:
        request.session.create()
        session_id = request.session.session_key
        request.session["session_id"] = session_id
    logging.info(f"session_id: {session_id}")

    if not question:
        profile_image_url = "/static/img/historical/historical_user.png"
        return render(
            request,
            "chatbot/historical_chatbot.html",
            {"profile_image_url": profile_image_url},
        )

    response = process_historical_chatbot_request(question, session_id, request.user)
    for step in response:
        logging.info(step)
    final_output = step["output"]
    recommended_titles = extract_titles(final_output)
    if recommended_titles:
        logging.info(recommended_titles)
        save_recommended_works(request.user, recommended_titles, "romance")
    return HttpResponse(markdown2.markdown(final_output))