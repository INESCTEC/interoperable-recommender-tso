<div align="center">
  <img src="/docs/images/logo.png"  align="middle">
</div>

-----------------------------------------------------

[![version](https://img.shields.io/badge/version-1.0.1-blue.svg)]()
[![status](https://img.shields.io/badge/status-production-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Image Size](https://img.shields.io/badge/image%20size-1.52GB-blue.svg)]()

-----------------------------------------------------

## Overview

The **Interoperable Recommender** is a data-driven solution aimed at enabling the participation of consumers in enhancing the resilience of the European energy infrastructure. This novel service harnesses the potential of innovative algorithms and leverages the publicly accessible ENTSO-E Transparency Platform to assess country-specific vulnerabilities related to loss of load and generation curtailment. 

The main goal is to enable energy applications to empower European citizens with actionable recommendations on a national level, encouraging adaptive energy consumption during periods of expected system vulnerability. The service provides day-ahead hourly recommendations, tailored to meet the unique needs of each country while accounting for interconnections within the broader European network.

> [!IMPORTANT]
> This software is currently providing recommendation actions for Interconnect [Wattchr](https://wattchr.eu/). See the official [booklet](docs/interconnect-booklet.pdf) for more details.

### News

* **2024-12:** Our work is now published on iScience! [See our publication](https://www.cell.com/iscience/fulltext/S2589-0042(24)02655-5).
* **2024-10:** The **Interoperable Recommender** was presented at Enlit 2024! [See our presentation](docs/ir-enlit-2024-presentation.pdf).

### Publication

You can find more details on the methodology and use cases of the Interoperable Recommender in our open-source publication, available in iScience, an interdisciplinary open-access journal.

#### [Read our paper](https://www.cell.com/iscience/fulltext/S2589-0042(24)02655-5)

### List of European Countries with Recommendations

Below you can find a list of countries for which recommendations can be generated (limited by data availability on ENTSO-E TP platform).

| country_code   | country_name           |
|:---------------|:-----------------------|
| AL             | Albania                |
| AT             | Austria                |
| BA             | Bosnia and Herzegovina |
| BE             | Belgium                |
| BG             | Bulgaria               |
| CH             | Switzerland            |
| CY             | Cyprus                 |
| CZ             | Czech Republic         |
| DE             | Germany                |
| DK             | Denmark                |
| EE             | Estonia                |
| ES             | Spain                  |
| FI             | Finland                |
| FR             | France                 |
| GB             | United Kingdom         |
| GR             | Greece                 |
| HR             | Croatia                |
| HU             | Hungary                |
| IE             | Ireland                |
| IT             | Italy                  |
| LT             | Lithuania              |
| LU             | Luxembourg             |
| LV             | Latvia                 |
| ME             | Montenegro             |
| MK             | North Macedonia        |
| MT             | Malta                  |
| NL             | Netherlands            |
| NO             | Norway                 |
| PL             | Poland                 |
| PT             | Portugal               |
| RO             | Romania                |
| RS             | Serbia                 |
| SE             | Sweden                 |
| SI             | Slovenia               |
| SK             | Slovakia               |


## Initial setup:

> [!WARNING]
> The following commands assume that you are running them from the root directory of the project (`energy_app/`)

### Configure environment variables:

The `dotenv` file provides a template for all the environment variables needed by this project. 
To configure the environment variables, copy the `dotenv` file to `.env` and fill in the values for each variable.

```shell
cp dotenv .env
```
> [!NOTE]
>  In windows, just copy-paste the `dotenv` file and rename it to `.env`.


The following environment variables are required.

| variable   | Type | description            |
|:-----------|:-----|:-----------------------|
| POSTGRES_HOST         | String  | Recommender Database Host. Defaults to 'postgresql', which will work if you deploy the recommender in the same server as its database (see `docker-compose.yml` file) |
| POSTGRES_DB         | String  | Recommender Database name. Defaults to `master`  |
| POSTGRES_USER         | String  | Recommender Database Username. Defaults to `postgres`. |
| POSTGRES_PASSWORD     | String  | Recommender Database Password. Defaults to `postgres`.  |
| POSTGRES_PORT         | Integer | Recommender Database Port. Defaults to `5432` |
| ENTSOE_API_KEY        | String  | ENTSO-E TP `API KEY` used for data retrieval. See the official  [ENTSO-E Transparency Platform RESTful API documentation](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html) for more information. |


Optional environment variables can be configured to send recommendations to the [Energy APP (Wattchr) RESTful API server](https://wattchr.eu/energy-app/api/v2/swagger-ui/index.html#/).

| variable             | Type    | description            |
|:---------------------|:--------|:-----------------------|
| POST_TO_ENERGY_APP   | Integer | If `1`, executes an HTTP POST to the `ENERGYAPP_HOST` REST API server. Defaults to 0 (no request) |
| POST_ONLY_ON_UPDATES | Integer | If `1` Only executes HTTP POST if there are updates in the number of hours with available recommendations. Defaults to 0 (always attempts to POST if `POST_TO_ENERGY_APP` = 1) |
| ENERGYAPP_N_RETRIES  | Integer | Number of retries in case of problems on the HTTP requests to the  `ENERGYAPP_HOST` REST API server. |
| ENERGYAPP_APIKEY     | String  | API KEY for `ENERGYAPP_HOST` REST API server |
| ENERGYAPP_BASEPATH   | String  | Base API Path for `ENERGYAPP_HOST` REST API server |

> [!NOTE]
> The `Interoperable Recommender` is able to execute independently of the integration with an external REST API server (e.g., Wattchr). The current (optional) integration was developed within [InterConnect Energy APP](https://interconnectproject.eu/energy-applications/) pilot, with the latter API being used as a central layer of authentication / communication uniformization with end-users, for the recommendations provided by the `Interoperable Recommender`.


### With Docker:

To launch the docker containers stack:

```shell
docker compose up -d
```

> [!IMPORTANT]
> This will launch the database container and the 'energy_app' container. Note that both database schema will be initialized and the database migrations will be applied.


Then, import the table schema to the database (set in the environment variables).

```shell
docker compose run --rm energy_app alembic upgrade head
```


Assure that the database fixtures are imported by running the following command (in some platforms there might be issues with the entrypoint):

```shell
docker compose run --rm energy_app python load_db_fixtures.py
```

### With Local Python Interpreter:

If you prefer using your local python interpreter (instead of Docker), you'll need to manually perform the installation steps. Meaning:
1. Install the python dependencies
   ```shell
   pip install -r requirements.txt
   ```
2. Start the database container
   ```shell
   docker compose up -d postgresql
   ```
3. Apply the database migrations
    ```shell
    alembic upgrade head
    ```
4. Run the 'load_db_fixtures.py' script to init the database with its fixtures
    ```shell
    python load_db_fixtures.py
    ```


### Getting your ENTSO-E API Key and preparing the database

The `Interoperable Recommender` needs at least 6 months of historical data to successfully execute. This means that, after a successful deployment, you will need to do an initial upload of historical data to its internal databases. 

You can quickly do this by using the `data acquisition` module, which retrieves data from the [ENTSO-E Transparency Platform RESTful API](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html). 

The following execution will retrieve data from multiple system variables (and every country) for the past 180 days.


```shell
docker compose run --rm energy_app python main_data_acquisition.py --lookback_days=180
```

> [!WARNING]
> This command will take a while to execute. The `ENTSOE_API_KEY` environment variable must be declared to authenticate in the ENTSO-E TP API.


### How to run:

> [!WARNING]
> The following instructions assume that the database is already initialized and the database migrations are already applied. If not, please refer to the [Initial setup](#initial-setup) section.


#### Scheduled Tasks

The `Interoperable Recommender` has two main types of scheduled tasks. 

1. Data acquistion task: Retrieve TSO adta from ENTSO-E TP platform
2. Recommender execution task: Create national level recommendations (recommendations output stored in `energy_app/files/operational`)

If you're using a Linux server, you can quickly import the scheduled tasks by updating your `crontab` with the information available on the directory `cron/project_crontab` of this project.


#### Data acquisition tasks:

To launch the data acquisition pipeline, execute the `main_data_acquisition.py` script from the `energy_app` container:

***With Docker:***

```shell
docker compose run --rm energy_app python main_data_acquisition.py
```

***With Local Python interpreter:***

```shell
python main_data_acquisition.py
```

This will trigger the entire process of data ETL. Namely:

  1. Data retrieval from ENTSO-E Transparency Platform (via it's [REST API](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)). By default, data is retrieved for the past 2 days and for the day ahead (forecasts).
  2. Data manipulation and transformation (e.g., aggregate data by control area for NTC forecasts)
  3. Data load to the central database (PostgreSQL)

> [!IMPORTANT] 
> We recommend you run the data acquisition pipeline with a 
> lookback period to minimize the amount of missing historical data. 
> For example, to run the data acquisition pipeline for the last 7 days, run the following command:
> ```shell
>   docker compose run --rm energy_app python main_data_acquisition.py --lookback_days=7
> ```

#### Risk assessment and recommendation creation task:

To launch the recommender main pipeline, execute the `main.py` script from the `energy_app` container:

***With Docker:***

```shell
docker compose run --rm energy_app python main.py
```

***With Local Python interpreter:***

```shell
python main.py
```

This will run the following operations pipeline:
  1. Create country load & generation quantile forecasts and load system dataset from the service database (i.e., raw TSO data)
  2. Calculate Risk-Reserve Curve and risk evaluation per country
  3. Create Risk Coordination Matrix and set the final risk level for each country
  4. Prepare the final JSON payload with the recommendations for each country (as required by the EnergyAPP backend)
     * Store JSON payload in `energy_app/files/operational` directory
  5. (Optional) Perform HTTP request to POST the recommendations to the EnergyAPP backend


An overview of the full pipeline is available in the image below (press to zoom).

<img src="docs/images/energy_app_pipeline.png" alt="drawing" width="800"/>


> [!IMPORTANT]
> This methodology depends on accurate probabilistic (quantile) forecasts created by internal quantile regression models, which also depend on the availability of historical data for country generation / load (actual and forecasted). Please run the data acquisition task for a minimum of 6 month lookback to assure a good forecast quality.



### Outputs:

Interoperable recommender provides, as output, hourly recommendations for all the active countries.
The recommendations are provided in JSON format and are available in the `energy_app/files/output/` directory after each execution.

The following outputs are available, per country:

| variable        | type   | description                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-----------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| drr             | float  | Deterministic rule for reserve (DRR)                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| reserve         | float  | Reserve capacity to meet the system risk threshold reserve   (of the risk-reserve capacity curve)                                                                                                                                                                                                                                                                                                                                                                     |
| origin          | string | <ul>      <li>"individual" - recommendation based on risk mitigation   need for this country</li>      <li>"interconnection" - recommendation for this country, to   help mitigate risk in neighbour country</li>      </ul>                                                                                                                                                                                                                                          |
| risk_evaluation | string | <ul>      <li>"not available' - no recommendation was issued (e.g., due to   insuficient data, or internal error)</li>      <li> 'healthy' - system not at risk of loss of load or energy   generation curtailment</li>      <li> 'increase' - system (or neighbour system) at risk and raised   need to increase energy consumption</li>      <li>'decrease' - system (or neighbour system) at risk and raised need   to decrease energy consumption</li>      </ul> |
| risk_level      | int    | Risk threshold magnitude (0- healthy, 1-Low, 2-Medium,   3-high, 4-very high)                                                                                                                                                                                                                                                                                                                                                                                         |

Besides storing this information in the local directory, it is currently also pushed to the [Wattchr backend](https://wattchr.eu/energy-app/api/v2/swagger-ui/index.html#/) via HTTP POST request.

### Database maintenance / management:

#### Table creation / migrations:

We use `alembic` library for database migrations. To create a new table, follow the steps below:

1. Add ORM SQLalchemy model in `energy_app/database/alembic/models.py` script
2. Create new revision with alembic:
```shell
alembic revision --autogenerate -m "Added ntc forecasts table"
```
3. Apply the new revision with alembic:
```shell
alembic upgrade head
```


#### Database Backup:

##### To .bak file:

This software includes a database management tool. Which backups the database to a local file. To run the backup script, execute the following command:

```shell
docker-compose -f docker-compose.yml run --rm energy_app python db_maintenance.py backup database --file_name=<file_path_here>
```

##### To CSVs:

Alternatively, the database can be backed up to CSV. To run the backup script, execute the following command:

```shell
docker-compose -f docker-compose.yml run --rm energy_app python db_maintenance.py backup table
```

> [!IMPORTANT]
> There are multiple backup options. You can check the available options via:
> ```shell
>docker-compose -f docker-compose.yml run --rm energy_app python db_maintenance.py backup --help
> ```

#### Database VACUUM:

Database optimization ops are also available (and frequently executed). To run a DB vacuum:

```shell
docker-compose -f docker-compose.yml run --rm energy_app python db_maintenance.py vacuum database
```

### Database full reset:

First, stop database container then remove the database volume and
start the database container again.

```shell
docker compose down postgresql  # Stops the database container
docker volume rm energy-app-recommender_postgresql-data  # Removes the database volume
docker compose up -d postgresql  # Starts the database container
```

Then, run the following command to apply the database migrations (with alembic):

***With Docker:***

```shell
docker compose run --rm energy_app alembic upgrade head
docker compose run --rm energy_app python load_db_fixtures.py
```

***With Local Python interpreter:***
```shell
alembic upgrade head
python load_db_fixtures.py
```

### CLI arguments:

The `energy_app` process pipeline can be triggered with the following CLI arguments:

```shell
python main.py --help
```

#### Examples:

To execute for a specific launch time:
    
```shell
python main.py --launch_time "2023-06-01T00:00:00Z"
```

To execute for a specific launch time and set a specific lookback period 
(in days) to retrieve historical data from the database 
(i.e., previously acquired via the ENTSO-E data acquisition module)

```shell
# Retrieve ENTSO-E data for the 30 days prior to 2023-06-01T00:00:00Z
python main.py --launch_time="2023-06-01T00:00:00Z" --lookback_days=30
```


## Contacts:

If you have any questions regarding this project, please contact the following people:

Developers (SW source code / methodology questions):
  - José Andrade <jose.r.andrade@inesctec.pt>
  - Carlos Silva <carlos.silva@inesctec.pt>
  - Carlos Pereira <carlos.m.pereira@inesctec.pt>
  - Igor Abreu <igor.c.abreu@inesctec.pt>

Contributors / Reviewers (methodology questions):
  - Ricardo Bessa <ricardo.j.bessa@inesctec.pt>
  - David Rua <david.e.rua@inesctec.pt>
