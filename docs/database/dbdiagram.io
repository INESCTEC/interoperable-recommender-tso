// Use DBML to define your database structure
// Docs: https://dbml.dbdiagram.io/docs

Table country {
  code varchar(4) [primary key]
  name text
  timezone text
  active boolean
  biggest_generator integer
  updated_at timestamp
}

Table country_neighbours {
  country_code varchar(4)
  neighbours varchar(100)
  updated_at timestamp
}


Table generation_forecast {
  country_code varchar(4)
  timestamp_utc timestamp
  value double
  unit text
  updated_at timestamp
}


Table load_actual {
  country_code varchar(4)
  timestamp_utc timestamp
  value double
  unit text
  updated_at timestamp
}


Table load_forecast {
  country_code varchar(4)
  timestamp_utc timestamp
  value double
  unit text
  updated_at timestamp
}


Table pump_load_forecast {
  country_code varchar(4)
  timestamp_utc timestamp
  value double
  unit text
  updated_at timestamp
}


Table res_generation_actual {
  country_code varchar(4)
  timestamp_utc timestamp
  value double
  unit text
  updated_at timestamp
}

Table res_generation_forecast {
  country_code varchar(4)
  timestamp_utc timestamp
  value double
  unit text
  updated_at timestamp
}

Table sce {
  from_country_code varchar(4)
  to_country_code varchar(4)
  timestamp_utc timestamp
  value double
  unit text
  updated_at timestamp
}


Ref: country.code > country_neighbours.country_code
Ref: country.code > generation_forecast.country_code
Ref: country.code > load_actual.country_code
Ref: country.code > load_forecast.country_code
Ref: country.code > pump_load_forecast.country_code
Ref: country.code > res_generation_actual.country_code
Ref: country.code > res_generation_forecast.country_code
Ref: country.code > sce.country_code
