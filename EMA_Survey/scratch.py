afternoon_data = []

afternoon_data.append('Start Time = ' + str(afternoon_ema_df['StartDate'].values))
afternoon_data.append('End Time = '+  str(afternoon_ema_df['EndDate'].values))

if (int(afternoon_ema_df['Break'].values)) > 0 and (int(afternoon_ema_df['Break'].values)) < 8:
    afternoon_data.append('Break = ' + (str(afternoon_ema_df['Break'].values)))
else:
    pass

if (int(afternoon_ema_df['Rushed'].values)) > 0 and (int(afternoon_ema_df['Rushed'].values)) < 8:
    afternoon_data.append('Rushed = ' + (str(afternoon_ema_df['Rushed'].values)))
else:
    pass

if (int(afternoon_ema_df['Confront_authority'].values)) > 0 and (int(afternoon_ema_df['Confront_authority'].values)) < 8:
    afternoon_data.append('Confront_authority = ' + (str(afternoon_ema_df['Confront_authority'].values)))
else:
    pass

if (int(afternoon_ema_df['Rude_family'].values)) > 0 and (int(afternoon_ema_df['Rude_family'].values)) < 8:
    afternoon_data.append('Rude_family = ' + (str(afternoon_ema_df['Rude_family'].values)))
else:
    pass

if (int(afternoon_ema_df['gen_disrespect'].values)) > 0 and (int(afternoon_ema_df['gen_disrespect'].values)) < 8:
    afternoon_data.append('gen_disrespect = ' + (str(afternoon_ema_df['gen_disrespect'].values)))
else:
    pass

if (int(afternoon_ema_df['COVID_concern'].values)) > 0 and (int(afternoon_ema_df['COVID_concern'].values)) < 8:
    afternoon_data.append('COVID_concern = ' + (str(afternoon_ema_df['COVID_concern'].values)))
else:
    pass

if (int(afternoon_ema_df['Discomfort'].values)) > 0 and (int(afternoon_ema_df['Discomfort'].values)) < 8:
    afternoon_data.append('Discomfort = ' + (str(afternoon_ema_df['Discomfort'].values)))
else:
    pass

if (int(afternoon_ema_df['Lack_support'].values)) > 0 and (int(afternoon_ema_df['Lack_support'].values)) < 8:
    afternoon_data.append('Lack_support = ' + (str(afternoon_ema_df['Lack_support'].values)))
else:
    pass

if (int(afternoon_ema_df['Team_value'].values)) > 0 and (int(afternoon_ema_df['Team_value'].values)) < 8:
    afternoon_data.append('Team_value = ' + (str(afternoon_ema_df['Team_value'].values)))
else:
    pass

if (int(afternoon_ema_df['Demands'].values)) > 0 and (int(afternoon_ema_df['Demands'].values)) < 8:
    afternoon_data.append('Demands = ' + (str(afternoon_ema_df['Demands'].values)))
else:
    pass

if (int(afternoon_ema_df['Death'].values)) > 0 and (int(afternoon_ema_df['Death'].values)) < 8:
    afternoon_data.append('Death = ' + (str(afternoon_ema_df['Death'].values)))
else:
    pass

if (int(afternoon_ema_df['Other_work-stress'].values)) > 0 and (int(afternoon_ema_df['Other_work-stress'].values)) < 8:
    afternoon_data.append('Other_work-stress = ' + (str(afternoon_ema_df['Other_work-stress'].values)))
else:
    pass

if (int(afternoon_ema_df['Other_non-work-stress'].values)) > 0 and (int(afternoon_ema_df['Other_non-work-stress'].values)) < 8:
    afternoon_data.append('Other_non-work-stress = ' + (str(afternoon_ema_df['Other_non-work-stress'].values)))
else:
    pass



    