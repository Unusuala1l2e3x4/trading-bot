print(do_longs, do_shorts)

from IPython.utils import io
results_list = []
for i in tqdm(list(np.arange(0.2, 4.01, 0.2))):

    with io.capture_output() as captured:
        params.times_buying_power = i

        balance, longs_executed, shorts_executed, balance_change, mean_profit_loss_pct, win_mean_profit_loss_pct, lose_mean_profit_loss_pct, winrate, total_costs, \
            avg_sub_pos, avg_transact, count_entry_adjust, count_entry_skip, count_exit_adjust, count_exit_skip = \
            TradingStrategy(TouchDetectionAreas.from_dict(touch_detection_areas), params).run_backtest()

        trades_executed = longs_executed + shorts_executed
        newrow = {'times_buying_power':f'{i:.1f}', 'balance_pt_change':[f'{balance_change:.4f}%'], 'balance':[f'{balance:.4f}'], 'trades_executed':[trades_executed], \
            'mean_profit_loss_pct':f'{mean_profit_loss_pct:.6f}', 'win_mean_profit_loss_pct':f'{win_mean_profit_loss_pct:.6f}', 'lose_mean_profit_loss_pct':f'{lose_mean_profit_loss_pct:.6f}',
            'winrate':f'{winrate:.6f}', \
            'total_costs':f'${total_costs:.4f}', \
            'avg num sub pos created':f'{avg_sub_pos:.2f}', 'avg num transactions':f'{avg_transact:.2f}',\
            'count_entry_adjust':f'{count_entry_adjust}', 'count_entry_skip':f'{count_entry_skip}',\
            'count_exit_adjust':f'{count_exit_adjust}', 'count_exit_skip':f'{count_exit_skip}'}
        results_list.append(pd.DataFrame.from_dict(newrow))

results = pd.concat(results_list,ignore_index=True)
results




# results.to_csv(f'times_buying_power_tests/{symbol}_times_buying_power_test_{start_date.split()[0]}_{end_date.split()[0]}.csv',index=False)