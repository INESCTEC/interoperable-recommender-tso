import argparse
import datetime as dt

datetime_parser_ = lambda s: dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')


def load_cli_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch_time", required=False, type=datetime_parser_,
                    help="Launch Time UTC [%Y-%m-%dT%H:%M:%SZ]",
                    default=dt.datetime.utcnow())
    ap.add_argument("--output_dir", required=False, type=str,
                    help="Output directory (path appended to energy_app/files/ dir)",
                    default="operational")
    args = vars(ap.parse_args())
    return args["launch_time"], args["output_dir"]


def load_cli_args_acquisition():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch_time", required=False, type=datetime_parser_,
                    help="Launch Time UTC [%Y-%m-%dT%H:%M:%SZ]",
                    default=dt.datetime.utcnow())
    ap.add_argument("--lookback_days", required=False, type=int,
                    help="Number of days to lookback when querying "
                         "ENTSO-E data from DB",
                    default=2)
    args = vars(ap.parse_args())
    return args["launch_time"], args["lookback_days"]
