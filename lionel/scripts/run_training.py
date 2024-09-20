import sys
import datetime as dt
from lionel.scripts.process_train_data import get_train
from lionel.db.connector import DBManager
from lionel.constants import DATA
from lionel.model.hierarchical import FPLPointsModel


def run(dbm, next_gw, season, sampler_config=None):
    data = get_train(dbm, season, next_gw)
    fplm = FPLPointsModel(sampler_config)
    fplm.fit(data, data.points)
    today = dt.datetime.today().strftime("%Y%m%d")
    fplm.save(DATA / f"analysis/hm_{today}.nc")
    return True


if __name__ == "__main__":
    dbm = DBManager(DATA / "fpl.db")
    sampler_config = {
        "draws": 1000,
        "tune": 200,
        "chains": 4,
    }
    next_gw, season = [int(x) for x in sys.argv[1:]]
    run(dbm, next_gw, season, sampler_config)
