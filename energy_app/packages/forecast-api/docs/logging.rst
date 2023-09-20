.. _logging_ref:

Logging
=======

You can access the messages of this module by initializing the `logging <https://docs.python.org/3/library/logging.html>`_ module in your main.py file as shown bellow:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)

    # available levels:
    # - INFO
    # - DEBUG
    # - WARNING
    # - ERROR

With the previous configuration you I'll see the messages displayed in the console with the default format.
However you can change the format of the message and timestamp. We recommend the following config:

.. code-block:: python

    import logging
    logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-50s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )

If needed you can define a file to export the logging, using the keyword `filename` in the `logging.basicConfig()`.
Furthermore you can have multiple loggers, for instance one with level.INFO which is shown in the console and another with level.DEBUG which is exported
to a file:

.. code-block:: python

    import os
    import logging

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # logger with level.DEBUG which is exported to 'logging_debug.log'
    logging.basicConfig(filename=os.path.join(dir_path, 'logging_debug.log'),
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-50s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )

    # logger with level.INFO which is shown in the console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(name)-50s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # when using 3rd party packages, sometimes it's useful to silent
    # (or change the level) their logging messages:
    logging.getLogger("cassandra").setLevel(logging.WARNING)