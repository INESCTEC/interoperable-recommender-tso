import scipy.interpolate
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from conf import settings

plot_flag = False


def calculate_risk(countries_qt_forecast):
    """Builds a risk curve for each country in the input list.

    This function takes a dictionary containing forecast and capacity data for each country and evaluate the risk level
    for each country.

    Parameters:
    :param countries_qt_forecast: dictionary with forecast and capacity data for each country
    :type countries_qt_forecast: dict
    
    Returns:
    :countries_risk: dictionary with 'upward' and 'downward' risks analysis for each country is presented, 'drr',
    'risk_evaluation', 'risk_level' are also included.
    :rtype: dict
    """
    countries_risk = dict()
    for country_code in countries_qt_forecast.keys():
        # get data for country
        country_forecasts = countries_qt_forecast[country_code]

        log_msg_ = f"[RiskReserve:{country_code}] Get quantiles ..."
        logger.debug(log_msg_)

        # quantiles for load and RES
        qt_names_load = country_forecasts['load']['total'].iloc[0].index
        qt_load = np.array([round(float(x.strip('q')) / 100, 3) for x in qt_names_load])
        qt_names_res = country_forecasts['generation']['renewable'].iloc[0].index
        qt_res = np.array([round(float(x.strip('q')) / 100, 3) for x in qt_names_res])

        # values for load and RES quantiles
        qt_load_values = country_forecasts['load']['total'].to_numpy()
        qt_res_values = country_forecasts['generation']['renewable'].to_numpy()
        logger.debug(f"{log_msg_} ... Ok!")

        log_msg_ = f"[RiskReserve:{country_code}] Calculate System Margin (SM) ..."
        logger.debug(log_msg_)
        system_margin = SM_computations()

        # compute system margin curve
        system_margin.all_uncertainty(country_code, qt_load_values, qt_load, qt_res_values, qt_res,
                                      saldo_intercon=np.zeros((country_forecasts['load']['total'].shape[0])))
        logger.debug(f"{log_msg_} ... Ok!")

        log_msg_ = f"[RiskReserve:{country_code}] Shift SM (+ Conventional Generation - Pumped Storage +- SCE) ..."
        logger.debug(log_msg_)
        # get Conventional Generation, Pumped Storage Consumption, and Scheduled Commercial Exchanges
        conventional_gen = country_forecasts['generation']['conventional']['cg_generation_forecast'].to_list()
        pumped_consumption = country_forecasts['load']['pump']['pump_load_forecast'].to_list()
        sce_export = country_forecasts['sce']['export'].copy()
        mininum_data = sce_export.groupby(sce_export.index).count()['from_country_code'][0]  # expected data points
        sce_export = sce_export.groupby(sce_export.index)['value'].sum(min_count=mininum_data).to_list()
        sce_import = country_forecasts['sce']['import'].copy()
        sce_import = sce_import.groupby(sce_import.index)['value'].sum(min_count=mininum_data).to_list()
        # shift system margin curve
        for hour in range(0, country_forecasts['load']['total'].shape[0]):
            # check if all interconnections are available
            sce_available = country_forecasts['data_validator']['sce'][hour]
            if not sce_available:
                system_margin.SM_list[hour].x = None
                missing_import = country_forecasts['data_validator']['sce_missing'][hour]['import']
                missing_export = country_forecasts['data_validator']['sce_missing'][hour]['export']
                if missing_import:
                    log_msg_ = (f"[RiskReserve:{country_code}][Hour:{hour}]: No data to shift System Margin... "
                                f"Missing SCE import data from countries {missing_import}")
                    logger.error(log_msg_)
                if missing_export:
                    log_msg_ = (f"[RiskReserve:{country_code}][Hour:{hour}]: No data to shift System Margin... "
                                f"Missing SCE export data from countries {missing_export}")
                    logger.error(log_msg_)
                continue

            # check for missing data and skip hour
            headers = ['Conventional Generation', 'Pumped Consumption', 'SCE Import', 'SCE Export']
            missing_data = [headers[i] for i, x in enumerate([conventional_gen[hour], pumped_consumption[hour],
                                                              sce_import[hour], sce_export[hour]]) if np.isnan(x)]
            if missing_data:
                system_margin.SM_list[hour].x = None
                log_msg_ = f"[RiskReserve:{country_code}][Hour:{hour}]: No data to shift System Margin... Missing {missing_data}"
                logger.error(log_msg_)
                continue

            if system_margin.SM_list[hour].x is not None:
                sce_total = sce_import[hour] - sce_export[hour]
                system_margin.SM_list[hour].x = system_margin.SM_list[hour].x + conventional_gen[hour] - pumped_consumption[hour] + sce_total
        logger.debug(f"{log_msg_} ... Ok!")

        # Compute risk-reserve margin curve
        log_msg_ = f"[RiskReserve:{country_code}] Calculate Risk-Reserve curve ..."
        logger.debug(log_msg_)

        reserves = reserve_level(
            country_code,
            system_margin.SM_list,
            index_name_up='LOLP', 
            index_value_up=settings.RISK_THRESHOLD,
            index_name_down='PWRE', 
            index_value_down=settings.RISK_THRESHOLD,
            )
        logger.debug(f"{log_msg_} ... Ok!")

        # Get evaluation risk for each timestep
        log_msg_ = f"[RiskReserve:{country_code}] Compare Reserve with DRR  ..."
        logger.debug(log_msg_)

        country_risk = get_risk_evaluation(country_code, country_forecasts, reserves, system_margin)
        countries_risk[country_code] = country_risk
        
        # logger.debug(f"[RiskReserve:{country_code}] {countries_risk[country_code]}")
        logger.debug(f"{log_msg_} ... Ok!")

        # Plot curves
        if plot_flag:
            plot_reserve(
                country_code,
                countries_qt_forecast[country_code]['load']['total'].shape[0], 
                qt_res, qt_load, 
                qt_res_values,
                qt_load_values, 
                system_margin, 
                reserves, 
            )

    return countries_risk


class rv:  # class discrete random variable
    def __init__(self, x, Fx, px):  # constructor for the random variable
        self.x = x
        self.Fx = Fx
        if px.size == 0:
            self.px = np.array([])
        else:
            self.px = px
            
    def minus(self, randv1, randv2, N):  # method for the subtraction of two random variables (the two arguments are the two random variables)
        randv2.x = -np.resize(randv2.x, [1 * randv2.x.size])  # the randv2 is the negative random variable
        randv2.Fx = 1-randv2.Fx  # the randv2 is the negative random variable
        randv1.x = np.resize(randv1.x, [1 * randv1.x.size])
        
        # possible max and min to filter the result in the end
        x_max = randv1.x.max() + randv2.x.max()  # max possible value
        x_min = randv1.x.min() + randv2.x.min()  # min possible value
        
        # lower and upper values for the grid
        lower = np.array([randv1.x.min(), randv2.x.min()]).min()
        upper = np.array([randv1.x.max(), randv2.x.max()]).max()
        
        # extend the interpolation minimum of the randv1 random variable
        delta = (randv1.x.min() -lower)/100.0
        xx = np.arange(lower, randv1.x.min(), delta)
        f_xx = np.repeat(0.0, xx.size)  # zeros for the new values in the cdf function
        randv1.Fx = np.concatenate((f_xx, randv1.Fx))  # merge the arrays
        randv1.x = np.concatenate((xx, randv1.x))
       
        # extend the interpolation maximum of the randv1 random variable
        xx = np.array([randv1.x.max() + 0.1])
        f_xx = np.array([1.0])
        randv1.Fx = np.concatenate((randv1.Fx, f_xx))  # merge the arrays
        randv1.x = np.concatenate((randv1.x, xx))
        
        # extend the interpolation minimum of the randv2 random variable
        xx = np.array([randv2.x.min()-0.1])  # only because the minimum and the lower value are the same
        f_xx = np.array([0.0])  # zeros for the new values in the cdf function
        randv2.Fx = np.concatenate((randv2.Fx, f_xx,))  # merge the arrays
        randv2.x = np.concatenate((randv2.x, xx))
        
        # extend the interpolation maximum of the randv2 random variable
        delta = (upper-randv2.x.max())/100.0
        xx = np.arange(randv2.x.max() + delta, upper + 2 * delta, delta)
        aux = -xx  # sort the values to merge correctely
        ord = aux.ravel().argsort()
        xx = xx[ord]
        f_xx = np.repeat(1.0, xx.size)  # ones for the new values in the cdf function
        randv2.Fx = np.concatenate((f_xx, randv2.Fx))  # merge the arrays
        randv2.x = np.concatenate((xx, randv2.x))
        ord = randv2.x.ravel().argsort()  # sort the values to be in increasing order
        randv2.x = randv2.x[ord]
        randv2.Fx = randv2.Fx[ord]
        
        # linear interpolation of the two CDF functions
        f_randv1 = scipy.interpolate.interp1d(randv1.x, randv1.Fx)
        f_randv2 = scipy.interpolate.interp1d(randv2.x, randv2.Fx)
        
        delta = (upper-lower)/N  # Width of the new grid
        grid = np.arange(lower, upper+delta, delta)
        if grid.size != (N+1):  # test if the length is above N because sometimes the function arange fails
            grid = grid[0:(grid.size-1)]
            
        # differentiation of the cdf function to compute the values of the pmf function
        p_randv1 = f_randv1(grid)
        p_randv2 = f_randv2(grid)
        pmf_randv1 = p_randv1[1:p_randv1.size]-p_randv1[0:p_randv1.size-1]
        pmf_randv2 = p_randv2[1:p_randv2.size]-p_randv2[0:p_randv2.size-1]
        
        new_lower = 2 * lower
        new_upper = 2 * upper
        x = np.arange(new_lower, new_upper+delta,delta)  # new grid
        if x.size != (N*2+1):  # test iof the length is above N because sometimes the function arange fails
            x = x[0:(x.size-1)]
        
        # prob vectors with dimension 2*N to perform convolution
        v1 = np.concatenate((pmf_randv1,np.array(np.repeat(0.0,N))))
        v2 = np.concatenate((pmf_randv2,np.array(np.repeat(0.0,N))))
            
        # FFT
        fft1 = fft(v1)
        fft2 = fft(v2)

        # FFT inverse (pmf of the subtraction)
        inv_fft = np.concatenate(([0.0], np.real(ifft(fft1*fft2))))  # inverse FFT
            
        #  cut the vector inv_fft between the min and the final max
        x_min_pos = int((x <= x_min).nonzero()[0][(x <= x_min).nonzero()[0].size-1])
        x_max_pos = int((x >= x_max).nonzero()[0][0])
        
        # save the result of the difference between the random variables
        self.px = inv_fft[x_min_pos:x_max_pos]  # pmf prob
        self.Fx = self.px.cumsum()  # cdf prob
        self.x = x[x_min_pos:x_max_pos]  # x


class SM_computations:  # class discrete random variable
    def __init__(self):  # constructor for the random variable
        self.SM_list = list()
        self.Gen_list = list()
        self.Load_list = list()
    
    def all_uncertainty(self, country_code, load_X, load_Fx, generation_X, generation_Fx, saldo_intercon):  # method for the subtraction of two random variables (the two arguments are the two random variables)
        for i in range(load_X.shape[0]):  # for each look-ahead time of day D+1
            L = rv(load_X[i, :], load_Fx, np.array([]))  # r.v. load
            
            G = rv(generation_X[i, :], generation_Fx, np.array([]))  # r.v. generation
            
            G.x = G.x + saldo_intercon[i]  # add NTC
                        
            SM = rv(0.0, 0.0, np.array([]))  # system margin r.v.

            try:
                SM.minus(G, L, pow(2, 14))  # system margin pmf
            except ValueError:
                # if minus function fails due to NaN
                SM.x, SM.Fx, SM.px = None, None, None
                # check what quantile forecast is missing
                headers = ['Generation', 'Load']
                missing_data = [headers[i] for i, x in enumerate([G.x, L.x]) if np.isnan(x).any()]
                log_msg_ = (f"[RiskReserve:{country_code}][Hour:{i}]: No data to calculate System Margin... "
                            f"Check {missing_data} quantile forecasts...")
                logger.error(log_msg_)
            
            self.Load_list.append(L)  # list with the rv load
            self.Gen_list.append(G)  # list with the rv generation
            self.SM_list.append(SM)  # list with the rv system margin


def reserve_level(
        country_code,
        SM,  # system margin after applying SM_computations() class
        index_name_up,  # possibilities: 'LOLP', 'LOLE [min/h]', 'EENS [MWh]'
        index_value_up,  # risk for up reserve: e.g.0.001 means 0.1%
        index_name_down,  # possibilities: 'PWRE', 'WREE [min/h]', 'ESE [MWh]'
        index_value_down,  # risk for down reserve: e.g.0.001 means 0.1%
        ):

    risk_curves_up = list()
    risk_curves_down = list()

    reserves_tested_up = list()
    reserves_tested_down = list()

    reserve_needs_up = np.array([])
    reserve_needs_down = np.array([])
    
    for i, sm in enumerate(SM):  # compute for each look-ahead time
        if sm.x is None:
            # continue to next hour if system margin is not available
            reserves_tested_up.append(None)
            reserves_tested_down.append(None)
            risk_curves_up.append(None)
            risk_curves_down.append(None)
            reserve_needs_up = np.hstack((reserve_needs_up, None))
            reserve_needs_down = np.hstack((reserve_needs_down, None))
            log_msg_ = f"[RiskReserve:{country_code}][Hour:{i}]: No data to calculate Reserve Level... Missing System Margin ..."
            logger.error(log_msg_)
            continue

        r_inc = 5
        index_up = np.array([]) 
        index_down = np.array([])
        Reserve_up = np.array([])
        Reserve_down = np.array([])

        R = 0.0
        while True:
            # build the relation function between the reserve level and the risk index
            new_SM_up = sm.x + float(R)
            Reserve_up = np.hstack((Reserve_up, float(R)))
            
            if index_name_up == 'LOLP':
                index_up = np.hstack((index_up, sm.px[(new_SM_up <= 0.0).nonzero()[0]].sum()))
            
            if index_name_up == 'LOLE [min/h]':
                index_up = np.hstack((index_up, sm.px[(new_SM_up <= 0.0).nonzero()[0]].sum()*60))
            
            if index_name_up == 'EENS [MWh]':
                prob = sm.px[(new_SM_up <= 0.0).nonzero()[0]]  # array with the probabilities of the negative margin
                ener = new_SM_up[(new_SM_up <= 0.0).nonzero()[0]]  # array with the energy of the negative margin
                index_up = np.hstack((index_up, abs((prob*ener).sum())))

            if index_up[-1] <= index_value_up:
                break

            R += r_inc

        if R != 0.0:
            reserves_tested_up.append(np.arange(0, R+r_inc, r_inc))
        else:
            reserves_tested_up.append(np.arange(0, r_inc, r_inc))

        R = 0.0
        while True:
            new_SM_down = sm.x - float(R)
            Reserve_down = np.hstack((Reserve_down, float(R)))

            if index_name_down == 'PWRE':
                index_down = np.hstack((index_down, sm.px[(new_SM_down > 0.0).nonzero()[0]].sum()))
            
            if index_name_down == 'WREE [min/h]':
                index_down = np.hstack((index_down, sm.px[(new_SM_down > 0.0).nonzero()[0]].sum()*60))
            
            if index_name_down == 'ESE [MWh]':
                prob = sm.px[(new_SM_down > 0.0).nonzero()[0]]  # array with the probabilities of the negative margin
                ener = new_SM_down[(new_SM_down > 0.0).nonzero()[0]]  # array with the energy of the negative margin
                index_down = np.hstack((index_down, abs((prob*ener).sum())))

            if index_down[-1] <= index_value_down:
                break
            R += r_inc
            
        if R != 0.0:
            reserves_tested_down.append(np.arange(0, R+r_inc, r_inc))
        else:
            reserves_tested_down.append(np.arange(0, r_inc, r_inc))
        
        risk_curves_up.append(index_up)
        risk_curves_down.append(index_down)
        
        try:
            ord = index_up.ravel().argsort()  # order the arrays
            func_up = scipy.interpolate.interp1d(index_up[ord], Reserve_up[ord])  # interpolation of the relation (Reserva/ Valor da medida de risco escolhida)

            reserve_needs_up = np.hstack((reserve_needs_up, func_up(index_value_up)))  # determination of the reserve accordindly with the reference risk
        except ValueError:
            reserve_needs_up = np.hstack((reserve_needs_up, 0.0))
            log_msg_ = f"[RiskReserve:{country_code}][Hour:{i}]: Failed Reserve Up Calculation ..."
            logger.error(log_msg_)

        try:
            ord = index_down.ravel().argsort()  # order the arrays
            func_down = scipy.interpolate.interp1d(index_down[ord], Reserve_down[ord])  # interpolation of the relation
            
            reserve_needs_down = np.hstack((reserve_needs_down, func_down(index_value_down)))  # determination of the reserve accordindly with the reference risk
        except ValueError:
            reserve_needs_down = np.hstack((reserve_needs_down, 0.0))
            log_msg_ = f"[RiskReserve:{country_code}][Hour:{i}]: Failed Reserve Down Calculation ..."
            logger.error(log_msg_)

    output = np.zeros((reserve_needs_up.shape[0], 2))
    output[:, 0] = reserve_needs_up
    output[:, 1] = reserve_needs_down

    out = {'reserves_for_selected_risks': output, 
           'reserves_tested_up': reserves_tested_up,
           'reserves_tested_down': reserves_tested_down,
           'discrete_risk_curves_up': risk_curves_up,
           'discrete_risk_curves_down': risk_curves_down}
    
    return out


def get_risk_evaluation(country_code, country_forecasts, reserves, system_margin):
    # reserves: reserves dict output of risk_curve 
    largest_unit_country = country_forecasts['generation']['biggest_gen_capacity']
    largest_pump_unit = country_forecasts['load']['max_pump_historical']
    if largest_pump_unit is None:
        # use largest unit as default if largest pumping is not available
        largest_pump_unit = country_forecasts['generation']['biggest_gen_capacity']

    country_risk = dict()
    for timestep in range(0, country_forecasts['load']['total'].shape[0]):
        # get reserve levels at risk threshold
        reserve_needs_up = reserves['reserves_for_selected_risks'][timestep][0]
        reserve_needs_down = reserves['reserves_for_selected_risks'][timestep][1]

        # continue to next hour if reserve values or largest unit is not available
        if np.isnan(reserve_needs_up) or np.isnan(reserve_needs_down) or largest_unit_country is None:
            # check what information is missing
            headers = ['Reserve Up', 'Reserve Down']
            missing_data = [headers[i] for i, x in enumerate([reserve_needs_up, reserve_needs_down]) if np.isnan(x)]
            if largest_unit_country is None:
                missing_data.append('Largest Unit')
            # prepare default output structure
            country_risk[timestep] = {
                'upward': {
                    'drr': None,
                    'reserve': None,
                    'risk_evaluation': 'not available',
                    'risk_level': None
                },
                'downward': {
                    'drr': None,
                    'reserve': None,
                    'risk_evaluation': 'not available',
                    'risk_level': None
                }
            }

            # add debug information
            country_risk[timestep]['debug'] = {
                'system_margin_x': None,
                'system_margin_y': None,
                'reserve_curve_up_x': None,
                'reserve_curve_up_y': None,
                'reserve_needs_up': reserve_needs_up,
                'reserve_curve_down_x': None,
                'reserve_curve_down_y': None,
                'reserve_needs_down': reserve_needs_down,
                'risk_threshold': settings.RISK_THRESHOLD
            }

            log_msg_ = f"[RiskReserve:{country_code}][Hour:{timestep}]: No data to calculate Risk Level... Missing {missing_data}"
            logger.error(log_msg_)
            continue

        # get MAPE of Generation Forecasts
        mape_gen = country_forecasts['generation']['gen_mape'] \
            if country_forecasts['generation']['gen_mape'] else 0
        gen_drr = country_forecasts['generation']['renewable']['q50'].iloc[timestep] * mape_gen
        # get MAPE of Load Forecasts
        mape_load = country_forecasts['load']['load_mape'] \
            if country_forecasts['load']['load_mape'] else 0
        load_drr = country_forecasts['load']['total']['q50'].iloc[timestep] * mape_load

        # Risk (upward)
        # calculate Deterministic Rule for Reserve (DRR)
        coeff_a = 10  # MW
        coeff_b = 150  # MW
        drr_up_secondary = np.sqrt(coeff_a * country_forecasts['load']['total']['q50'].max() + coeff_b ** 2) - coeff_b
        drr_up_tertiary = largest_unit_country + gen_drr + load_drr
        drr_up_country = drr_up_secondary + drr_up_tertiary

        # If R <= DRR -> Healthy
        if reserve_needs_up <= drr_up_country:
            evaluation_risk_up = 'healthy'
            risk_level_up = 0
        # Else, decrease consumption is indicated:
        else: 
            evaluation_risk_up = 'decrease'
            drr_minus_reserve_up = drr_up_country - reserve_needs_up
            # Level #1 - upper 0 to 20% of drr_up_country - reserve_needs_up
            if np.abs(drr_minus_reserve_up) <= 0.2 * largest_unit_country:
                risk_level_up = 1

            # Level #2 - upper 20% to 40% of drr_up_country - reserve_needs_up
            elif 0.2 * largest_unit_country < np.abs(drr_minus_reserve_up) <= 0.4 * largest_unit_country:
                risk_level_up = 2
                
            # Level #3 - upper 40% to 80% of drr_up_country - reserve_needs_up
            elif 0.4 * largest_unit_country < np.abs(drr_minus_reserve_up) <= 0.8 * largest_unit_country:
                risk_level_up = 3

            # Level #4 - upper 80% to drr_up_country - reserve_needs_up
            elif np.abs(drr_minus_reserve_up) > 0.8 * largest_unit_country:
                risk_level_up = 4
            # Else, no risk level is indicated
            else:
                risk_level_up = None

        # Risk (downward)
        # Calculate Deterministic Rule for Reserve (DRR)
        # todo: temporary fix (use same reference as upward reserve)
        #  exception for PT and ES
        #  remove to revert to previous logic
        if country_code not in ['ES', 'PT']:
            largest_pump_unit = largest_unit_country
        # --
        drr_down_secondary = np.sqrt(coeff_a * country_forecasts['load']['total']['q50'].max() + coeff_b ** 2) - coeff_b
        drr_down_tertiary = largest_pump_unit + gen_drr + load_drr
        drr_down_country = drr_down_secondary + drr_down_tertiary

        # If R <= DRR -> Healthy
        if reserve_needs_down <= drr_down_country:
            evaluation_risk_down = 'healthy'
            risk_level_down = 0
        # Else, increase consumption is indicated:
        else:
            evaluation_risk_down = 'increase'
            drr_minus_reserve_down = drr_down_country - reserve_needs_up
            # Level #1 - upper 0 to 20% of drr_up_country - reserve_needs_up
            if np.abs(drr_minus_reserve_down) <= 0.2 * largest_pump_unit:
                risk_level_down = 1

            # Level #2 - upper 20% to 40% of drr_up_country - reserve_needs_up
            elif 0.2 * largest_pump_unit < np.abs(drr_minus_reserve_down) <= 0.4 * largest_pump_unit:
                risk_level_down = 2

            # Level #3 - upper 40% to 80% of drr_up_country - reserve_needs_up
            elif 0.4 * largest_pump_unit < np.abs(drr_minus_reserve_down) <= 0.8 * largest_pump_unit:
                risk_level_down = 3

            # Level #4 - upper 80% to drr_up_country - reserve_needs_up
            elif np.abs(drr_minus_reserve_down) > 0.8 * largest_pump_unit:
                risk_level_down = 4
            # Else, no risk level is indicated
            else:
                risk_level_down = None

        country_risk[timestep] = {
            'upward': {
                'drr': drr_up_country,
                'reserve': reserve_needs_up,
                'risk_evaluation': evaluation_risk_up,
                'risk_level': risk_level_up
            },
            'downward': {
                'drr': drr_down_country,
                'reserve': reserve_needs_down,
                'risk_evaluation': evaluation_risk_down,
                'risk_level': risk_level_down
            }
        }

        # add debug information
        country_risk[timestep]['debug'] = {
            'system_margin_x': system_margin.SM_list[timestep].x.tolist(),
            'system_margin_y': system_margin.SM_list[timestep].px.tolist(),
            'reserve_curve_up_x': reserves['reserves_tested_up'][timestep].tolist(),
            'reserve_curve_up_y': reserves['discrete_risk_curves_up'][timestep].tolist(),
            'reserve_needs_up': reserve_needs_up,
            'reserve_curve_down_x': reserves['reserves_tested_down'][timestep].tolist(),
            'reserve_curve_down_y': reserves['discrete_risk_curves_down'][timestep].tolist(),
            'reserve_needs_down': reserve_needs_down,
            'risk_threshold': settings.RISK_THRESHOLD
        }

    return country_risk


def plot_reserve(country, timestep_ahead, qt_gen, qt_load, qt_gen_values, qt_load_values, system_margin, reserves):

    max_subplot = 4
    font = {'family': 'arial',
            'size': 6}

    plt.rc('font', **font)

    if timestep_ahead > max_subplot:
        num_figures = timestep_ahead // max_subplot
    else:
        num_figures = 1

    for figure_num in range(num_figures):
        num_rows = min(max_subplot, timestep_ahead - figure_num * max_subplot)
        if num_rows <= 0:
            break

        figure, axis = plt.subplots(num_rows, 5)

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.5)

        for i, timestep in enumerate(range(figure_num * 4, min((figure_num + 1) * 4, timestep_ahead))):
            # empty plots if reserves for that timestep are not available
            if not np.isnan(reserves['reserves_for_selected_risks'][timestep][0]) and not \
                    np.isnan(reserves['reserves_for_selected_risks'][timestep][1]):
                axis[i, 0].plot(qt_gen_values[timestep, :], qt_gen, 'o', markersize=1)
                axis[i, 0].set_title("Gen Q")
                axis[i, 0].set_ylim([0, None])

                axis[i, 1].plot(qt_load_values[timestep, :], qt_load, 'o', markersize=1)
                axis[i, 1].set_title("Load Q")
                axis[i, 1].set_ylim([0, None])

                axis[i, 2].plot(system_margin.SM_list[timestep].x, system_margin.SM_list[timestep].px, 'o', markersize=1)
                axis[i, 2].set_title("SM ")
                axis[i, 2].set_ylim([0, None])

                axis[i, 3].plot(reserves['reserves_tested_up'][timestep], reserves['discrete_risk_curves_up'][timestep], 'o', markersize=1)
                axis[i, 3].set_title("Risk (LOLP)")
                axis[i, 3].plot(reserves['reserves_for_selected_risks'][timestep][0], settings.RISK_THRESHOLD, 'o', markersize=8, color='red')
                axis[i, 3].axhline(y=settings.RISK_THRESHOLD, color='black', linestyle='--', linewidth=0.5)
                axis[i, 3].axvline(x=reserves['reserves_for_selected_risks'][timestep][0], color='black', linestyle='--', linewidth=0.5)
                axis[i, 3].set_ylim([0, None])

                axis[i, 4].plot(reserves['reserves_tested_down'][timestep], reserves['discrete_risk_curves_down'][timestep], 'o', markersize=1)
                axis[i, 4].plot(reserves['reserves_for_selected_risks'][timestep][1], settings.RISK_THRESHOLD, 'o', markersize=8, color='red')
                axis[i, 4].axhline(y=settings.RISK_THRESHOLD, color='black', linestyle='--', linewidth=0.5)
                axis[i, 4].axvline(x=reserves['reserves_for_selected_risks'][timestep][1], color='black', linestyle='--', linewidth=0.5)
                axis[i, 4].set_title("Risk (PWRE)")
                axis[i, 4].set_ylim([0, None])

        plt.savefig(f'{country}_{figure_num}.pdf')
