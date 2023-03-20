import logging
import os

logger = logging.getLogger()

def print_args(args, output_dir=None):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")

    if output_dir is not None:
        with open(os.path.join(output_dir, "args.txt"), "w") as f:
            for key, val in sorted(vars(args).items()):
                keystr = "{}".format(key) + (" " * (30 - len(key)))
                f.write(f"{keystr}   {val}\n")