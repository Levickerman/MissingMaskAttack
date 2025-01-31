"""
Configure logging here.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from tsdb.utils.logging import Logger

# initialize a logger for PyPOTS logging
logger_creator = Logger(name="PyPOTS running log")
# logger_creator.set_saving_path("results", "mma_log")
logger = logger_creator.logger
