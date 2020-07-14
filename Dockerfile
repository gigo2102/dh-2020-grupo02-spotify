FROM python:3
ENV APP /app
ENV PORT 8090
RUN mkdir $APP
WORKDIR $APP
EXPOSE 8090
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN sh setup.sh
ENTRYPOINT [ "streamlit", "run", "app.py" ]