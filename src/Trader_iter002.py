from typing import List, Any
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

import json
from typing import Dict, List

import json
from typing import Dict, List

class Trader:
    def bid(self):
        return 15

    def run(self, state: TradingState):
        """
        Pure market-making strategy:
        - EMERALDS fair value = constant 10000
        - TOMATOES fair value = rolling mean of last 5 mids
        - TOMATOES z-score uses rolling std of last 50 mids
        """

        # ---------- config ----------
        POSITION_LIMITS = {
            "TOMATOES": 20,
            "EMERALDS": 20,
        }

        EMERALDS_FAIR = 10000

        # Tight quoting parameters
        EMERALDS_HALF_SPREAD = 1
        TOMATOES_BASE_HALF_SPREAD = 1

        # Tomatoes z-score thresholds
        TOMATOES_TIGHT_Z = 0.35
        TOMATOES_SKEW_Z = 1.00

        # Inventory skew
        INV_SKEW_PER_UNIT = 0.05

        # ---------- load persistent state ----------
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except Exception:
                trader_state = {}
        else:
            trader_state = {}

        price_history = trader_state.get("mid_history", {
            "TOMATOES": [],
            "EMERALDS": [],
        })

        result: Dict[str, List[Order]] = {}
        conversions = 0

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            if product not in POSITION_LIMITS:
                result[product] = orders
                continue

            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                result[product] = orders
                continue

            # Best prices
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            # Save history
            if product not in price_history:
                price_history[product] = []
            price_history[product].append(mid_price)

            # Keep only recent values
            if len(price_history[product]) > 60:
                price_history[product] = price_history[product][-60:]

            position = state.position.get(product, 0)
            pos_limit = POSITION_LIMITS[product]
            buy_capacity = pos_limit - position
            sell_capacity = pos_limit + position

            # ---------- EMERALDS: constant mean ----------
            if product == "EMERALDS":
                fair_value = EMERALDS_FAIR
                half_spread = EMERALDS_HALF_SPREAD

                inv_skew = INV_SKEW_PER_UNIT * position
                bid_quote = int(round(fair_value - half_spread - inv_skew))
                ask_quote = int(round(fair_value + half_spread - inv_skew))

                bid_quote = min(bid_quote, best_ask - 1)
                ask_quote = max(ask_quote, best_bid + 1)

                if buy_capacity > 0:
                    orders.append(Order(product, bid_quote, min(5, buy_capacity)))
                if sell_capacity > 0:
                    orders.append(Order(product, ask_quote, -min(5, sell_capacity)))

            # ---------- TOMATOES: rolling mean + z-score ----------
            elif product == "TOMATOES":
                hist = price_history[product]

                # rolling fair = mean of last 5 mids
                last5 = hist[-5:]
                fair_value = sum(last5) / len(last5)

                # rolling std = std of last 50 mids
                last50 = hist[-50:]
                if len(last50) >= 2:
                    mean50 = sum(last50) / len(last50)
                    var50 = sum((x - mean50) ** 2 for x in last50) / len(last50)
                    std50 = var50 ** 0.5
                else:
                    std50 = 1.0

                if std50 < 1e-6:
                    std50 = 1.0

                zscore = (mid_price - fair_value) / std50
                inv_skew = INV_SKEW_PER_UNIT * position

                if abs(zscore) <= TOMATOES_TIGHT_Z:
                    # symmetric tight quoting
                    bid_quote = int(round(fair_value - TOMATOES_BASE_HALF_SPREAD - inv_skew))
                    ask_quote = int(round(fair_value + TOMATOES_BASE_HALF_SPREAD - inv_skew))

                    bid_quote = min(bid_quote, best_ask - 1)
                    ask_quote = max(ask_quote, best_bid + 1)

                    if buy_capacity > 0:
                        orders.append(Order(product, bid_quote, min(5, buy_capacity)))
                    if sell_capacity > 0:
                        orders.append(Order(product, ask_quote, -min(5, sell_capacity)))

                elif zscore > 0:
                    # price rich vs fair -> lean to selling
                    bid_quote = int(round(fair_value - 2 - inv_skew))
                    ask_quote = int(round(fair_value - inv_skew))

                    bid_quote = min(bid_quote, best_ask - 1)
                    ask_quote = max(ask_quote, best_bid + 1)

                    if abs(zscore) >= TOMATOES_SKEW_Z:
                        if sell_capacity > 0:
                            orders.append(Order(product, ask_quote, -min(8, sell_capacity)))
                    else:
                        if buy_capacity > 0:
                            orders.append(Order(product, bid_quote, min(2, buy_capacity)))
                        if sell_capacity > 0:
                            orders.append(Order(product, ask_quote, -min(6, sell_capacity)))

                else:
                    # price cheap vs fair -> lean to buying
                    bid_quote = int(round(fair_value - inv_skew))
                    ask_quote = int(round(fair_value + 2 - inv_skew))

                    bid_quote = min(bid_quote, best_ask - 1)
                    ask_quote = max(ask_quote, best_bid + 1)

                    if abs(zscore) >= TOMATOES_SKEW_Z:
                        if buy_capacity > 0:
                            orders.append(Order(product, bid_quote, min(8, buy_capacity)))
                    else:
                        if buy_capacity > 0:
                            orders.append(Order(product, bid_quote, min(6, buy_capacity)))
                        if sell_capacity > 0:
                            orders.append(Order(product, ask_quote, -min(2, sell_capacity)))

            result[product] = orders

        # ---------- persist state ----------
        traderData = json.dumps({
            "mid_history": price_history
        }, separators=(",", ":"))

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData