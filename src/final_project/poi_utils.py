def fraction( poi_messages, all_messages ):
    if poi_messages=="NaN":
        fraction=0
    else:
        if all_messages=="NaN":
            fraction=0
        else:
            fraction=float(1.0*poi_messages/all_messages)
    return fraction
