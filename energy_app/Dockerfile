# Use an official Python runtime as a parent image
FROM python:3.10-buster

#Install base software:
RUN apt-get update && apt-get -y install nano htop telnet lsb-release && apt-get clean all
# Add repo for older psql client
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" | tee  /etc/apt/sources.list.d/pgdg.list
RUN apt update && apt -y install postgresql-client-12

## Add the wait script to the image
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait /wait
RUN chmod +x /wait

# File Author / Maintainer
LABEL maintainer="jose.r.andrade@inesctec.pt"

# Set the working directory to /app
WORKDIR /energy_app

# Copying the requirements for installation to take
# advantage of the caching.
COPY energy_app/requirements.txt .
RUN pip install -r requirements.txt

# entsoe-py package installation:
COPY energy_app/packages /energy_app/packages
RUN cd /energy_app/packages/entsoe-py && python setup.py install
RUN cd /energy_app/packages/forecast-api && python setup.py install

# Launch the wait tool to wait for postgres to be up (check-compose)
CMD /wait

COPY ./energy_app /energy_app

RUN chmod +x entrypoint.sh
RUN ["sh","/energy_app/entrypoint.sh"]
#CMD ["tail", "-f", "/dev/null"]
