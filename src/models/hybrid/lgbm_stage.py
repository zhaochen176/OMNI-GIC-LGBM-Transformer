import lightgbm as lgb


def train_lgbm(
    X_train,
    y_train,
    X_valid,
    y_valid,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=int(num_boost_round),
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=False)],
    )
    return model
