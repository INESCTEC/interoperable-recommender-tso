-- schema.sql
\c master;


-- Country Info
CREATE TABLE IF NOT EXISTS public.country
(
    code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    name text COLLATE pg_catalog."default" NOT NULL,
    timezone text COLLATE pg_catalog."default" NOT NULL,
    active boolean NOT NULL,
    biggest_generator integer NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT country_pkey PRIMARY KEY (code),
    CONSTRAINT country_name_key UNIQUE (name)
)
ALTER TABLE public.country OWNER to postgres;


-- ENTSOE TP - Country Neighbours
CREATE TABLE IF NOT EXISTS public.country_neighbours
(
    country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    neighbours character varying(100) COLLATE pg_catalog."default",
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT country_neighbours_country_code_key UNIQUE (country_code),
    CONSTRAINT country_neighbours_country_code_fkey FOREIGN KEY (country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
ALTER TABLE public.country_neighbours OWNER to postgres;

-- ENTSOE TP - Generation Forecast
CREATE TABLE IF NOT EXISTS public.generation_forecast
(
    country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    timestamp_utc timestamp without time zone NOT NULL,
    value double precision,
    unit text COLLATE pg_catalog."default" NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT generation_forecast_country_code_timestamp_utc_key UNIQUE (country_code, timestamp_utc),
    CONSTRAINT generation_forecast_country_code_fkey FOREIGN KEY (country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
ALTER TABLE public.generation_forecast OWNER to postgres;

-- ENTSOE TP - Load Actual
CREATE TABLE IF NOT EXISTS public.load_actual
(
    country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    timestamp_utc timestamp without time zone NOT NULL,
    value double precision,
    unit text COLLATE pg_catalog."default" NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT load_actual_country_code_timestamp_utc_key UNIQUE (country_code, timestamp_utc),
    CONSTRAINT load_actual_country_code_fkey FOREIGN KEY (country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
ALTER TABLE public.load_actual OWNER to postgres;

-- ENTSOE TP - Load Forecast
CREATE TABLE IF NOT EXISTS public.load_forecast
(
    country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    timestamp_utc timestamp without time zone NOT NULL,
    value double precision,
    unit text COLLATE pg_catalog."default" NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT load_forecast_country_code_timestamp_utc_key UNIQUE (country_code, timestamp_utc),
    CONSTRAINT load_forecast_country_code_fkey FOREIGN KEY (country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)

ALTER TABLE public.load_forecast OWNER to postgres;

-- ENTSOE TP - Pump Load Forecast
CREATE TABLE IF NOT EXISTS public.pump_load_forecast
(
    country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    timestamp_utc timestamp without time zone NOT NULL,
    value double precision,
    unit text COLLATE pg_catalog."default" NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT pump_load_forecast_country_code_timestamp_utc_key UNIQUE (country_code, timestamp_utc),
    CONSTRAINT pump_load_forecast_country_code_fkey FOREIGN KEY (country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
ALTER TABLE public.pump_load_forecast OWNER to postgres;

-- ENTSOE TP - RES Generation Actual
CREATE TABLE IF NOT EXISTS public.res_generation_actual
(
    country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    timestamp_utc timestamp without time zone NOT NULL,
    value double precision,
    unit text COLLATE pg_catalog."default" NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT res_generation_actual_country_code_timestamp_utc_key UNIQUE (country_code, timestamp_utc),
    CONSTRAINT res_generation_actual_country_code_fkey FOREIGN KEY (country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
ALTER TABLE public.res_generation_actual OWNER to postgres;

-- ENTSOE TP - RES Generation Forecast
CREATE TABLE IF NOT EXISTS public.res_generation_forecast
(
    country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    timestamp_utc timestamp without time zone NOT NULL,
    value double precision,
    unit text COLLATE pg_catalog."default" NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT res_generation_forecast_country_code_timestamp_utc_key UNIQUE (country_code, timestamp_utc),
    CONSTRAINT res_generation_forecast_country_code_fkey FOREIGN KEY (country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
ALTER TABLE public.res_generation_forecast OWNER to postgres;


-- ENTSOE TP - Scheduled Commercial Exchanges
CREATE TABLE IF NOT EXISTS public.sce
(
    from_country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    to_country_code character varying(4) COLLATE pg_catalog."default" NOT NULL,
    timestamp_utc timestamp without time zone NOT NULL,
    value double precision,
    unit text COLLATE pg_catalog."default" NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    CONSTRAINT sce_from_country_code_to_country_code_timestamp_utc_key UNIQUE (from_country_code, to_country_code, timestamp_utc),
    CONSTRAINT sce_from_country_code_fkey FOREIGN KEY (from_country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT sce_to_country_code_fkey FOREIGN KEY (to_country_code)
        REFERENCES public.country (code) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)
ALTER TABLE public.sce OWNER to postgres;