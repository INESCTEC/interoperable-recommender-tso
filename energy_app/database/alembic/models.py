# coding: utf-8
from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Table, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Country(Base):
    __tablename__ = 'country'

    code = Column(String(4), primary_key=True)
    name = Column(Text, nullable=False, unique=True)
    timezone = Column(Text, nullable=False)
    active = Column(Boolean, nullable=False)
    biggest_gen_capacity = Column(Integer, nullable=True)
    biggest_gen_name = Column(Text, nullable=True)
    updated_at = Column(DateTime, nullable=False)


t_country_neighbours = Table(
    'country_neighbours', metadata,
    Column('country_code', ForeignKey('country.code'), nullable=False),
    Column('neighbours', String(100)),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('country_code')
)

t_generation_forecast = Table(
    'generation_forecast', metadata,
    Column('country_code', ForeignKey('country.code'), nullable=False),
    Column('timestamp_utc', DateTime, nullable=False),
    Column('value', Float(53)),
    Column('unit', Text, nullable=False),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('country_code', 'timestamp_utc')
)


t_load_actual = Table(
    'load_actual', metadata,
    Column('country_code', ForeignKey('country.code'), nullable=False),
    Column('timestamp_utc', DateTime, nullable=False),
    Column('value', Float(53)),
    Column('unit', Text, nullable=False),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('country_code', 'timestamp_utc')
)


t_load_forecast = Table(
    'load_forecast', metadata,
    Column('country_code', ForeignKey('country.code'), nullable=False),
    Column('timestamp_utc', DateTime, nullable=False),
    Column('value', Float(53)),
    Column('unit', Text, nullable=False),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('country_code', 'timestamp_utc')
)


t_pump_load_forecast = Table(
    'pump_load_forecast', metadata,
    Column('country_code', ForeignKey('country.code'), nullable=False),
    Column('timestamp_utc', DateTime, nullable=False),
    Column('value', Float(53)),
    Column('unit', Text, nullable=False),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('country_code', 'timestamp_utc')
)


t_res_generation_actual = Table(
    'res_generation_actual', metadata,
    Column('country_code', ForeignKey('country.code'), nullable=False),
    Column('timestamp_utc', DateTime, nullable=False),
    Column('value', Float(53)),
    Column('unit', Text, nullable=False),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('country_code', 'timestamp_utc')
)


t_res_generation_forecast = Table(
    'res_generation_forecast', metadata,
    Column('country_code', ForeignKey('country.code'), nullable=False),
    Column('timestamp_utc', DateTime, nullable=False),
    Column('value', Float(53)),
    Column('unit', Text, nullable=False),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('country_code', 'timestamp_utc')
)


t_sce = Table(
    'sce', metadata,
    Column('from_country_code', ForeignKey('country.code'), nullable=False),
    Column('to_country_code', ForeignKey('country.code'), nullable=False),
    Column('timestamp_utc', DateTime, nullable=False),
    Column('value', Float(53)),
    Column('unit', Text, nullable=False),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('from_country_code', 'to_country_code', 'timestamp_utc')
)


t_ntc = Table(
    'ntc_forecast', metadata,
    Column('from_country_code', ForeignKey('country.code'), nullable=False),
    Column('to_country_code', ForeignKey('country.code'), nullable=False),
    Column('timestamp_utc', DateTime, nullable=False),
    Column('value', Float(53)),
    Column('unit', Text, nullable=False),
    Column('updated_at', DateTime, nullable=False),
    UniqueConstraint('from_country_code', 'to_country_code', 'timestamp_utc')
)

t_report = Table(
    'report', metadata,
    Column('country_code', ForeignKey('country.code'), nullable=False),
    Column('table_entsoe', String, nullable=False),
    Column('day', Date, nullable=False),
    Column('max_created_at', DateTime, nullable=False),
    Column('row_count', Integer, nullable=False),
    UniqueConstraint('country_code', 'table_entsoe', 'day')
)

