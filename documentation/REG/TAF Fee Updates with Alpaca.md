

Info found in:
- https://docs.alpaca.markets/docs/broker-api-faq#trade-settlement
  - https://alpaca.markets/blog/reg-taf-fees/

# REG/TAF Fee Updates with Alpaca
By Hitoshi Harada • May 03, 2021

## TL;DR
We’re excited to announce some enhancements that went into effect April 26, 2021. First, based on feedback from our customers, the REG/TAF fee calculation has changed for all Alpaca customers. Second, for greater clarity and transparency, these fees are now listed as a separate line item in the account activity records on both the dashboard or through the API on any sales transactions.

## A little bit about the REG/TAF Fee
Every time you sell your stocks, the SEC and FINRA charge fees on the sale. The SEC charges \$8  for every \$1,000,000 (or \$0.000008 per dollar), and FINRA charges \$166 for every 1,000,000 shares (or \$0.000166 per share) sold up to a maximum of \$8.30 per trade. We denote the former as REG fee, and the latter as TAF fee. These fees are used by regulators to help offset the costs associated with regulating the equities market.

Regulators assess these fees on broker-dealers who report transactions. Broker-dealers, in turn, are allowed to pass these fees on to customers. Alpaca currently passes these fees on to customers who effect a sale transaction on a daily basis at the end of each trading day.

## Why does it matter?
In February 2021, Alpaca migrated to a new clearing broker partner, Velox. As part of this migration, Alpaca transitioned to its own back-office system that was built in-house. This transition included a change to how the REG/TAF fees were calculated on sale transactions. Below is an explanation of the difference; we call the first method a “Daily-consolidated” calculation and the second method a “Per-transaction” calculation.

## “Daily-consolidated” vs “Per-transaction”
Let’s say, you sold 100 shares of XYZ at $ \$100 $ per share. Your proceeds are \$10,000. The REG fee is \$0.08 $ (10,000 * 8.00/ 1,000,000) $. In order to book this value in the accounting system and charge it appropriately, we need to account for the $0.001 (0.1 cents). Alpaca currently rounds up to the nearest cent for any fractional cents.

For a small number of sales transactions per day, this “Per-transaction” approach has a negligible impact on customers. However, a potential problem arises when you have a lot of sales transactions in a day. Let’s say you had 100 sales of 1 share each for $100 per share. If we use the “Daily-consolidated” approach, then the REG fee will be:

$$ roundup(100 * 8 / 1,000,000 * 100) = roundup(\$0.08) = \$0.08 $$

but if we take the “Per-transaction” approach,

$$ roundup(100 * 8 / 1,000,000) * 100 = roundup(\$0.0008) * 100 = \$1.00 $$

As you can see, the only difference is the timing when you round up the fee amount. When we transitioned to our own back-office system in February 2021, we began using the “Per-transaction” method to calculate the REG/TAF fee. This resulted in some customers being assessed a larger than normal REG fee.

When we were made aware of this issue, we conducted a thorough analysis of our internal accounting methods, revisited our bookkeeping model, and assessed the impact to our operations.

## What changed and where we are going
Effective April 26th, the calculation has switched to the “Daily-consolidated” model. In parallel with this change, you should find a new activity record in your dashboard and the account activities API with the activity type called FEE every day you have a sales transaction. This should provide greater clarity with regard to the REG/TAF fees assessed on your sales transactions Please note that you may find two records for each REG and TAF if you traded with fractional shares. This is because what it is payable to is different, the street side transaction vs Alpaca’s inventory management transaction.

Of course, the biggest change with this switch is that you should see fewer charges in total as the unnecessary roundups were skipped.

As a commission-free broker surrounded by a great trader and dedicated developer community, we always appreciate your feedback. While this change wasn’t easy for us, your feedback led us in the right direction. Please give us your feedback any time you find something!