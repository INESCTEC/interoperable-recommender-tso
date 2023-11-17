<div align="center">
  <img src="/docs/images/logo.png"  align="middle">
</div>

-----------------------------------------------------

[![version](https://img.shields.io/badge/version-0.0.1-blue.svg)]()
[![status](https://img.shields.io/badge/status-production-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Image Size](https://img.shields.io/badge/image%20size-1.71GB-blue.svg)]()

Preliminary documentation available at the project `docs/` directory.

> **_NOTE:_** This software is currently providing recommendation actions for Interconnect [Wattchr](https://wattchr.eu/).


## Initial setup:

> **_NOTE:_**  The commands below assume that you are running them from the root directory of the project (`energy_app/`)

### Configure environment variables:

The `dotenv` file provides a template for all the environment variables needed by this project. 
To configure the environment variables, copy the `dotenv` file to `.env` and fill in the values for each variable.

```shell
   $ cp dotenv .env
```
**_NOTE:_** In windows, just copy-paste the `dotenv` file and rename it to `.env`.


### With Docker:

To launch the docker containers stack:

```shell
   $ docker compose up -d
```

**_NOTE:_**  This will launch the database container and the 'energy_app' container. Note that both database schema will be initialized and the database migrations will be applied.
**_NOTE:_**  The entrypoint of 'energy_app' container also runs the 'load_db_fixtures.py' script which fills the 'country' and 'country_neighbours' tables with data from the 'database/fixtures/countries.json' file.


Assure that the database fixtures are imported by running the following command (in some platforms there might be issues with the entrypoint):

```shell
   $ docker compose run --rm energy_app python load_db_fixtures.py
```

### With Local Python Interpreter:

If you prefer using your local python interpreter (instead of docker), you'll need to manually perform the installation steps. Meaning:
1. Install the python dependencies
   ```shell
        $ pip install -r requirements.txt
     ```
2. Start the database container
    ```shell
        $ docker compose up -d timescaledb
    ```
3. Apply the database migrations
    ```shell
        $ alembic upgrade head
    ```
4. Run the 'load_db_fixtures.py' script to init the database with its fixtures
    ```shell
        $ python load_db_fixtures.py
    ```

### How to run:

> **_NOTE 1:_**  The following instructions assume that the database is already initialized
and the database migrations are already applied. If not, please refer to the [Initial setup](#initial-setup) section.

> **_NOTE 2:_**  The commands below assume that you are running them from the root directory of the project (`energy_app/`)

#### Data acquisition tasks:

To launch the data acquisition pipeline, execute the `main_data_acquisition.py` script from the `energy_app` container:

***With Docker:***

```shell
   $ docker compose run --rm energy_app python main_data_acquisition.py
```

***With Local Python interpreter:***

```shell
   $ python main_data_acquisition.py
```

This will trigger the entire process of data ETL. Namely:

  1. Data retrieval from ENTSO-E Transparency Platform (via it's [REST API](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html))
  2. Data manipulation and transformation (e.g., aggregate data by control area for NTC forecasts)
  3. Data load to the central database (PostgreSQL / TimescaleDB)

> **_IMPORTANT:_** We recommend you run the data acquisition pipeline with a 
> lookback period to minimize the amount of missing historical data. 
> For example, to run the data acquisition pipeline for the last 30 days, run the following command:
> ```shell
>   $ docker compose run --rm energy_app python main_data_acquisition.py --lookback_days=7
> ```

#### Risk assessment and recommendation creation task:

To launch the recommender main pipeline, execute the `main.py` script from the `energy_app` container:

***With Docker:***

```shell
   $ docker compose run --rm energy_app python main.py
```

***With Local Python interpreter:***

```shell
   $ python main.py
```

This will run the following operations pipeline:
  1. Create country load & generation quantile forecasts and load system dataset from the service database (i.e., raw ENTSO-E data)
  2. Calculate Risk-Reserve Curve and risk evaluation per country
  3. Create Risk Coordination Matrix and set the final risk level for each country
  4. Prepare the final JSON payload with the recommendations for each country (as required by the EnergyAPP backend)
  5. Perform HTTP request to POST the recommendations to the EnergyAPP backend

An overview of the full pipeline is available in the image below (press to zoom).

<img src="docs/images/energy_app_pipeline.png" alt="drawing" width="800"/>


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

Besides storing this information in the local directory, it is currently also pushed to the EnergyAPP backend via HTTP POST request.

### Database maintenance / management:

#### Table creation / migrations:

We use `alembic` library for database migrations. To create a new table, follow the steps below:

1. Add ORM SQLalchemy model in `energy_app/database/alembic/models.py` script
2. Create new revision with alembic:
```shell
   $ alembic revision --autogenerate -m "Added ntc forecasts table"
```
3. Apply the new revision with alembic:
```shell
   $ alembic upgrade head
```


#### Database Backup:

##### To .bak file:

This software includes a database management tool. Which backups the database to a local file. To run the backup script, execute the following command:

```shell
   $ docker-compose -f docker-compose.prod.yml run --rm energy_app python db_maintenance.py backup database --file_name=<file_path_here>
```

##### To CSVs:

Alternatively, the database can be backed up to CSV. To run the backup script, execute the following command:

```shell
   $ docker-compose -f docker-compose.prod.yml run --rm energy_app python db_maintenance.py backup table
```

> **_NOTE:_** There are multiple backup options. You can check the available options via:
> ```shell
>  $ docker-compose -f docker-compose.prod.yml run --rm energy_app python db_maintenance.py backup --help
> ```

#### Database VACUUM:

Database optimization ops are also available (and frequently executed). To run a DB vacuum:

```shell
   $ docker-compose -f docker-compose.prod.yml run --rm energy_app python db_maintenance.py vacuum database
```

### Database full reset:

First, stop database container then remove the database volume and
start the database container again.

```shell
   $ docker compose down postgresql  # Stops the database container
   $ docker volume rm energy-app-recommender_postgresql-data  # Removes the database volume
   $ docker compose up -d postgresql  # Starts the database container
```

Then, run the following command to apply the database migrations (with alembic):

***With Docker:***

```shell
   $ docker compose run --rm energy_app sh entrypoint.sh
```

***With Local Python interpreter:***
```shell
   $ alembic upgrade head
   $ python load_db_fixtures.py
```

### CLI arguments:

The `energy_app` process pipeline can be triggered with the following CLI arguments:

```shell
   $ python main.py --help
```

#### Examples:

To execute for a specific launch time:
    
```shell
  $ python main.py --launch_time "2023-06-01T00:00:00Z"
```

To execute for a specific launch time and set a specific lookback period 
(in days) to retrieve historical data from the database 
(i.e., previously acquired via the ENTSO-E data acquisition module)

```shell
  # Retrieve ENTSOE data for the 30 days prior to 2023-06-01T00:00:00Z
  $ python main.py --launch_time "2023-06-01T00:00:00Z" --lookback_days 30
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

