import datetime
import numpy as np
import pandas as pd
from loguru import logger


def compute_actions(countries_details, countries_forecasts, countries_risk):
    """
    This function calculates the final actions for each country, considering their individual actions and the
    interconnection capacity between a country and their neighbors

    Parameters:
    :param countries_details: dictionary with details for each country
    :param countries_forecasts: dictionary with forecast and capacity data for each country
    :param countries_risk: dictionary with individual risk assessment results

    Returns:
    :final_actions_df: DataFrame with final actions to implement in each country (increase, decrease load)
    """
    active_countries = list(countries_risk.keys())

    # initialize output structure
    final_actions_df = pd.DataFrame(index=list(countries_risk.keys()))
    for timestep in range(0, list(countries_forecasts.items())[0][1]['timestamp_utc'].__len__()):
        # initialize country needs
        country_needs = np.zeros([countries_forecasts.keys().__len__(), countries_forecasts.keys().__len__()])
        country_can_help = np.zeros([countries_forecasts.keys().__len__(), countries_forecasts.keys().__len__()])

        for country_i, country_code in enumerate(active_countries):
            log_msg_ = f"[RiskCoordination:{country_code}][Hour:{timestep}] Checking risk ..."
            logger.debug(log_msg_)

            # check active interconnected countries
            interconnections = countries_details[country_code]['neighbours']
            active_interconnections = list(set(interconnections).intersection(list(countries_risk.keys())))

            # check if risk was computed for this hour in country i
            if countries_risk[country_code][timestep]['upward']['risk_evaluation'] == 'not available':
                log_msg_ = f"[RiskCoordination:{country_code}][Hour:{timestep}] Risk information not available..."
                logger.warning(log_msg_)
                continue

            # Get country risk and associated reserves
            if countries_risk[country_code][timestep]['upward']['risk_evaluation'] != 'healthy':
                # country i has upward risk
                reserve = countries_risk[country_code][timestep]['upward']['reserve']
                drr = countries_risk[country_code][timestep]['upward']['drr']
                action = countries_risk[country_code][timestep]['upward']['risk_evaluation']
                country_needs[country_i][country_i] = -1
                country_can_help[country_i][country_i] = 1
            elif countries_risk[country_code][timestep]['downward']['risk_evaluation'] != 'healthy':
                # country i has downward risk
                reserve = countries_risk[country_code][timestep]['downward']['reserve']
                drr = countries_risk[country_code][timestep]['downward']['drr']
                action = countries_risk[country_code][timestep]['downward']['risk_evaluation']
                country_needs[country_i][country_i] = 1
                country_can_help[country_i][country_i] = 1
            else:
                # country i does not need action from itself or neighboring countries
                log_msg_ = f"[RiskCoordination:{country_code}][Hour:{timestep}] No risk identified ..."
                logger.debug(log_msg_)
                continue

            log_msg_ = f"[RiskCoordination:{country_code}][Hour:{timestep}] Checking risk ... Ok!"
            logger.debug(log_msg_)

            log_msg_ = f"[RiskCoordination:{country_code}][Hour:{timestep}] Coordinating with interconnections ..."
            logger.debug(log_msg_)
            for interconnected_country in active_interconnections:
                # check if risk was computed for this hour for interconnected country j
                country_j = active_countries.index(interconnected_country)
                if countries_risk[interconnected_country][timestep]['upward']['risk_evaluation'] == 'not available':
                    log_msg_ = f"[RiskCoordination:{country_code}|{interconnected_country}][Hour:{timestep}] " \
                               f"Risk information not available..."
                    logger.warning(log_msg_)
                    continue

                # check if country i is import from or exporting to interconnected country j
                imports_from_interconnected = countries_forecasts[country_code]['sce']['import'][
                    countries_forecasts[country_code]['sce']['import']['from_country_code'] == interconnected_country][
                    'value'].iloc[timestep]
                exports_to_interconnected = countries_forecasts[country_code]['sce']['export'][
                    countries_forecasts[country_code]['sce']['export']['to_country_code'] == interconnected_country][
                    'value'].iloc[timestep]
                # calculate net import/export for interconnection ij
                sce_current = imports_from_interconnected - exports_to_interconnected

                # get Net Transfer Capacity (NTC) data for interconnection ij (import and export)
                ntc_data = countries_forecasts[country_code]['ntc']
                ntc_export = ntc_data['export'][ntc_data['export']['to_country_code'] == interconnected_country][
                    'value'].iloc[timestep]
                ntc_import = ntc_data['import'][ntc_data['import']['from_country_code'] == interconnected_country][
                    'value'].iloc[timestep]
                if np.isnan(ntc_export) or np.isnan(ntc_import):
                    # skip this interconnection due to missing data
                    log_msg_ = f"[RiskCoordination:{country_code}|{interconnected_country}][Hour:{timestep}] " \
                               f"Missing NTC data..."
                    logger.warning(log_msg_)
                    continue

                # what action should (if possible) the interconnected country j implement
                # calculate new scheduled commercial exchanges with (R-DRR) for interconnection ij
                if action == 'decrease':
                    country_needs[country_i][country_j] = -1
                    sce_new = sce_current + (reserve - drr)
                elif action == 'increase':
                    country_needs[country_i][country_j] = 1
                    sce_new = sce_current - (reserve - drr)

                # check NTC limit for export/import
                if sce_new < 0:
                    country_can_help[country_i][country_j] = 1 if np.abs(sce_new) <= np.abs(ntc_export) else 0
                elif sce_new > 0:
                    country_can_help[country_i][country_j] = 1 if np.abs(sce_new) <= np.abs(ntc_import) else 0

            log_msg_ = f"[RiskCoordination:{country_code}][Hour:{timestep}] Coordinating with interconnections ... Ok!"
            logger.debug(log_msg_)

        # multiply matrix with needed actions and possible actions to get feasible actions
        needs_df = pd.DataFrame(country_needs, index=list(countries_risk.keys()),
                                columns=list(countries_risk.keys()))
        can_help_df = pd.DataFrame(country_can_help, index=list(countries_risk.keys()),
                                   columns=list(countries_risk.keys()))
        feasible_actions = needs_df.mul(can_help_df)

        # give priority for individual actions
        final_actions = np.zeros(len(list(countries_risk.keys())))
        for country_j, column in enumerate(feasible_actions):
            column_values = feasible_actions[column].values
            if column_values[country_j] != 0:
                for country_i, value in enumerate(column_values):
                    # eliminate cross-actions (i =/= j) if country has individual action
                    column_values[country_i] = 0 if country_i != country_j else column_values[country_i]
                feasible_actions[country_j] = column_values
            # calculate most frequent action using sum of feasible actions by column
            final_actions[country_j] = sum(feasible_actions.to_numpy()[:, country_j])
            if final_actions[country_j] != 0:
                final_actions[country_j] = final_actions[country_j]/np.abs(final_actions[country_j])
        # insert final actions for this timestep in final dataframe
        final_actions_df[timestep] = final_actions

    # include countries with no forecast available
    active_countries_ = [country for country in countries_details.keys() if countries_details[country]['active']]
    final_actions_df = final_actions_df.reindex(active_countries_, fill_value=0)

    log_msg_ = f"[RiskCoordination] Final actions: \n {final_actions_df}"
    logger.debug(log_msg_)

    return final_actions_df


def prepare_output_structure(country_code, country_details, countries_qt_forecast, country_risks, country_actions):
    """"
    Function to prepare output structure to be sent by HTTP POST
    """

    timesteps = list(countries_qt_forecast[country_code]['timestamp_utc'])
    output = {
            'metadata': {
                'country_code': country_code,
                'country_name': country_details[country_code]['name'],
                'created_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'created_by': "interoperable-recommender-inesctec",
                'period': {
                    'start_datetime': timesteps[0].strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'end_datetime': timesteps[-1].strftime('%Y-%m-%dT%H:%M:%SZ'),
                }
            }
    }

    risk_data = []
    for i, timestep in enumerate(timesteps):
        # check risk data
        risk_up = country_risks[i]['upward']
        risk_down = country_risks[i]['downward']
        if risk_up['risk_evaluation'] != 'not available' and risk_down['risk_evaluation'] != 'not available':
            if risk_up['risk_evaluation'] != 'healthy':
                # country has upward risk
                risk_evaluation = risk_up['risk_evaluation']
                reserve = risk_up['reserve']
                drr = risk_up['drr']
                risk_level = risk_up['risk_level']
                origin = 'individual'
            elif risk_down['risk_evaluation'] != 'healthy':
                # country has downward risk
                risk_evaluation = risk_down['risk_evaluation']
                reserve = risk_down['reserve']
                drr = risk_down['drr']
                risk_level = risk_down['risk_level']
                origin = 'individual'
            else:
                # country has no risk (but can increase/decrease due to interconnections)
                drr = None
                reserve = None
                if country_actions.iloc[i] == 0:
                    # if no action is requested
                    risk_evaluation = 'healthy'
                    risk_level = 0
                    origin = 'individual'
                else:
                    # action is required by at least one interconnection
                    final_action = country_actions.iloc[i]
                    risk_evaluation = 'increase' if final_action > 0 else 'decrease'
                    risk_level = np.abs(final_action)
                    origin = 'interconnection'
        else:
            # risk not available
            drr = None
            reserve = None
            risk_evaluation = 'not available'
            risk_level = None
            origin = None

        risk = {
            'datetime': timestep.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'drr': drr,
            'reserve': reserve,
            'risk_evaluation': risk_evaluation,
            'risk_level': risk_level,
            'origin': origin
        }
        risk_data.append(risk)
    output.update({'data': risk_data})

    return output
