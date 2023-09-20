"""initial schema

Revision ID: f96b3115f4ab
Revises: 
Create Date: 2023-07-11 11:55:05.172783

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f96b3115f4ab'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('country',
    sa.Column('code', sa.String(length=4), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('timezone', sa.Text(), nullable=False),
    sa.Column('active', sa.Boolean(), nullable=False),
    sa.Column('biggest_generator', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('code'),
    sa.UniqueConstraint('name')
    )
    op.create_table('country_neighbours',
    sa.Column('country_code', sa.String(length=4), nullable=False),
    sa.Column('neighbours', sa.String(length=100), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['country_code'], ['country.code'], ),
    sa.UniqueConstraint('country_code')
    )
    op.create_table('generation_forecast',
    sa.Column('country_code', sa.String(length=4), nullable=False),
    sa.Column('timestamp_utc', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(precision=53), nullable=True),
    sa.Column('unit', sa.Text(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['country_code'], ['country.code'], ),
    sa.UniqueConstraint('country_code', 'timestamp_utc')
    )
    op.create_table('load_actual',
    sa.Column('country_code', sa.String(length=4), nullable=False),
    sa.Column('timestamp_utc', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(precision=53), nullable=True),
    sa.Column('unit', sa.Text(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['country_code'], ['country.code'], ),
    sa.UniqueConstraint('country_code', 'timestamp_utc')
    )
    op.create_table('load_forecast',
    sa.Column('country_code', sa.String(length=4), nullable=False),
    sa.Column('timestamp_utc', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(precision=53), nullable=True),
    sa.Column('unit', sa.Text(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['country_code'], ['country.code'], ),
    sa.UniqueConstraint('country_code', 'timestamp_utc')
    )
    op.create_table('pump_load_forecast',
    sa.Column('country_code', sa.String(length=4), nullable=False),
    sa.Column('timestamp_utc', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(precision=53), nullable=True),
    sa.Column('unit', sa.Text(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['country_code'], ['country.code'], ),
    sa.UniqueConstraint('country_code', 'timestamp_utc')
    )
    op.create_table('res_generation_actual',
    sa.Column('country_code', sa.String(length=4), nullable=False),
    sa.Column('timestamp_utc', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(precision=53), nullable=True),
    sa.Column('unit', sa.Text(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['country_code'], ['country.code'], ),
    sa.UniqueConstraint('country_code', 'timestamp_utc')
    )
    op.create_table('res_generation_forecast',
    sa.Column('country_code', sa.String(length=4), nullable=False),
    sa.Column('timestamp_utc', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(precision=53), nullable=True),
    sa.Column('unit', sa.Text(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['country_code'], ['country.code'], ),
    sa.UniqueConstraint('country_code', 'timestamp_utc')
    )
    op.create_table('sce',
    sa.Column('from_country_code', sa.String(length=4), nullable=False),
    sa.Column('to_country_code', sa.String(length=4), nullable=False),
    sa.Column('timestamp_utc', sa.DateTime(), nullable=False),
    sa.Column('value', sa.Float(precision=53), nullable=True),
    sa.Column('unit', sa.Text(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['from_country_code'], ['country.code'], ),
    sa.ForeignKeyConstraint(['to_country_code'], ['country.code'], ),
    sa.UniqueConstraint('from_country_code', 'to_country_code', 'timestamp_utc')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('sce')
    op.drop_table('res_generation_forecast')
    op.drop_table('res_generation_actual')
    op.drop_table('pump_load_forecast')
    op.drop_table('load_forecast')
    op.drop_table('load_actual')
    op.drop_table('generation_forecast')
    op.drop_table('country_neighbours')
    op.drop_table('country')
    # ### end Alembic commands ###
