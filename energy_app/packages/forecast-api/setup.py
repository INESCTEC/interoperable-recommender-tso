from setuptools import setup

PANDAS_MIN_VERSION = '0.20.3'
NUMPY_MIN_VERSION = '1.13.1'

setup(
    name='forecast_api',
    version='1.0',
    packages=['forecast_api', 'forecast_api.util', 'forecast_api.tests',
              'forecast_api.models',
              'forecast_api.models.algorithms',
              'forecast_api.models.algorithms.linear',
              'forecast_api.models.algorithms.ensemble',
              'forecast_api.models.algorithms.lazy_algorithms',
              'forecast_api.models.optimization',
              'forecast_api.models.optimization.metrics',
              'forecast_api.models.optimization.opt_algorithms',
              'forecast_api.models.optimization.opt_algorithms.bayesian_opt',
              'forecast_api.dataset',
              'forecast_api.evaluation'],
    url='',
    license='MIT',
    author='Jorge Filipe, Jose Andrade, Marisa Reis, João Viana',
    author_email="""
    -jorge.m.filipe@inesctec.pt
    -jose.r.andrade@inesctec.pt
    -marisa.m.reis@inesctec.pt
    -joao.p.viana@inesctec.pt
""",
    description='This Forecasting API establishes a simple framework for '
                'multiple forecasting tasks with the '
                'possibility of integrating several and distinct forecasting '
                'models without changing its core.'
                'Version Changelog: Included DST lags (optional).',
    install_requires=[
        'pandas>={0}'.format(PANDAS_MIN_VERSION),
        'numpy>={0}'.format(NUMPY_MIN_VERSION)
    ],
    zip_safe=False,
)

# python setup.py sdist --formats=gztar
