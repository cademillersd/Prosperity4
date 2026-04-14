from typing import List, Any, Dict, Optional
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


class Trader:
    def bid(self):
        return 15

    # =========================
    # Helper methods
    # =========================
    def get_mid_price(self, order_depth):
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return best_bid, best_ask, (best_bid + best_ask) / 2.0

    def update_price_history(self, price_history: Dict[str, List[float]], product: str, mid_price: float, max_len: int = 60):
        if product not in price_history:
            price_history[product] = []
        price_history[product].append(mid_price)
        if len(price_history[product]) > max_len:
            price_history[product] = price_history[product][-max_len:]

    def get_position_info(self, state, product: str, pos_limit: int):
        position = state.position.get(product, 0)
        buy_capacity = pos_limit - position
        sell_capacity = pos_limit + position
        return position, buy_capacity, sell_capacity

    def clamp_quotes(self, bid_quote: int, ask_quote: int, best_bid: int, best_ask: int):
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)
        return bid_quote, ask_quote

    def rolling_mean(self, values: List[float], window: int) -> float:
        chunk = values[-window:] if len(values) >= window else values
        return sum(chunk) / len(chunk)

    def rolling_std(self, values: List[float], window: int, fallback: float = 1.0) -> float:
        chunk = values[-window:] if len(values) >= window else values
        if len(chunk) < 2:
            return fallback
        mu = sum(chunk) / len(chunk)
        var = sum((x - mu) ** 2 for x in chunk) / len(chunk)
        std = var ** 0.5
        return std if std > 1e-6 else fallback

    # =========================
    # Strategy 1:
    # Strict market making with constant mean
    # =========================
    def market_make_constant_mean(
        self,
        product: str,
        state,
        order_depth,
        fair_value: float,
        pos_limit: int,
        half_spread: int = 1,
        lot_size: int = 5,
        inv_skew_per_unit: float = 0.05,
    ) -> List["Order"]:
        orders: List[Order] = []

        mid_data = self.get_mid_price(order_depth)
        if mid_data is None:
            return orders
        best_bid, best_ask, _ = mid_data

        position, buy_capacity, sell_capacity = self.get_position_info(state, product, pos_limit)

        inv_skew = inv_skew_per_unit * position
        bid_quote = int(round(fair_value - half_spread - inv_skew))
        ask_quote = int(round(fair_value + half_spread - inv_skew))

        bid_quote, ask_quote = self.clamp_quotes(bid_quote, ask_quote, best_bid, best_ask)

        if buy_capacity > 0:
            orders.append(Order(product, bid_quote, min(lot_size, buy_capacity)))
        if sell_capacity > 0:
            orders.append(Order(product, ask_quote, -min(lot_size, sell_capacity)))

        return orders

    # =========================
    # Strategy 2:
    # Z-score market making
    # - if mean_param is None -> floating mean
    # - else -> use supplied mean
    # =========================
    def market_make_zscore(
        self,
        product: str,
        state,
        order_depth,
        price_history: Dict[str, List[float]],
        pos_limit: int,
        mean_param: Optional[float] = None,
        mean_window: int = 5,
        std_window: int = 50,
        base_half_spread: int = 1,
        tight_z: float = 0.35,
        skew_z: float = 1.0,
        inv_skew_per_unit: float = 0.05,
        tight_size: int = 5,
        strong_size: int = 8,
        weak_buy_size: int = 2,
        weak_sell_size: int = 2,
        medium_buy_size: int = 6,
        medium_sell_size: int = 6,
    ) -> List["Order"]:
        orders: List[Order] = []

        mid_data = self.get_mid_price(order_depth)
        if mid_data is None:
            return orders
        best_bid, best_ask, mid_price = mid_data

        hist = price_history.get(product, [])
        if len(hist) == 0:
            return orders

        position, buy_capacity, sell_capacity = self.get_position_info(state, product, pos_limit)

        fair_value = mean_param if mean_param is not None else self.rolling_mean(hist, mean_window)
        std_value = self.rolling_std(hist, std_window, fallback=1.0)

        zscore = (mid_price - fair_value) / std_value
        inv_skew = inv_skew_per_unit * position

        if abs(zscore) <= tight_z:
            bid_quote = int(round(fair_value - base_half_spread - inv_skew))
            ask_quote = int(round(fair_value + base_half_spread - inv_skew))
            bid_quote, ask_quote = self.clamp_quotes(bid_quote, ask_quote, best_bid, best_ask)

            if buy_capacity > 0:
                orders.append(Order(product, bid_quote, min(tight_size, buy_capacity)))
            if sell_capacity > 0:
                orders.append(Order(product, ask_quote, -min(tight_size, sell_capacity)))

        elif zscore > 0:
            # price above fair -> lean short
            bid_quote = int(round(fair_value - 2 - inv_skew))
            ask_quote = int(round(fair_value - inv_skew))
            bid_quote, ask_quote = self.clamp_quotes(bid_quote, ask_quote, best_bid, best_ask)

            if abs(zscore) >= skew_z:
                if sell_capacity > 0:
                    orders.append(Order(product, ask_quote, -min(strong_size, sell_capacity)))
            else:
                if buy_capacity > 0:
                    orders.append(Order(product, bid_quote, min(weak_buy_size, buy_capacity)))
                if sell_capacity > 0:
                    orders.append(Order(product, ask_quote, -min(medium_sell_size, sell_capacity)))

        else:
            # price below fair -> lean long
            bid_quote = int(round(fair_value - inv_skew))
            ask_quote = int(round(fair_value + 2 - inv_skew))
            bid_quote, ask_quote = self.clamp_quotes(bid_quote, ask_quote, best_bid, best_ask)

            if abs(zscore) >= skew_z:
                if buy_capacity > 0:
                    orders.append(Order(product, bid_quote, min(strong_size, buy_capacity)))
            else:
                if buy_capacity > 0:
                    orders.append(Order(product, bid_quote, min(medium_buy_size, buy_capacity)))
                if sell_capacity > 0:
                    orders.append(Order(product, ask_quote, -min(weak_sell_size, sell_capacity)))

        return orders

    # =========================
    # Strategy 3:
    # Mean-reverting market making
    # More direct "buy cheap / sell rich" around a mean
    # =========================
    def market_make_mean_reverting(
        self,
        product: str,
        state,
        order_depth,
        price_history: Dict[str, List[float]],
        pos_limit: int,
        mean_param: Optional[float] = None,
        mean_window: int = 10,
        entry_threshold: float = 1.0,
        base_half_spread: int = 1,
        aggression: int = 1,
        lot_size: int = 6,
        inv_skew_per_unit: float = 0.05,
    ) -> List["Order"]:
        orders: List[Order] = []

        mid_data = self.get_mid_price(order_depth)
        if mid_data is None:
            return orders
        best_bid, best_ask, mid_price = mid_data

        hist = price_history.get(product, [])
        if len(hist) == 0:
            return orders

        position, buy_capacity, sell_capacity = self.get_position_info(state, product, pos_limit)

        fair_value = mean_param if mean_param is not None else self.rolling_mean(hist, mean_window)
        deviation = mid_price - fair_value
        inv_skew = inv_skew_per_unit * position

        # Neutral region: symmetric MM
        if abs(deviation) < entry_threshold:
            bid_quote = int(round(fair_value - base_half_spread - inv_skew))
            ask_quote = int(round(fair_value + base_half_spread - inv_skew))

            bid_quote, ask_quote = self.clamp_quotes(bid_quote, ask_quote, best_bid, best_ask)

            if buy_capacity > 0:
                orders.append(Order(product, bid_quote, min(lot_size, buy_capacity)))
            if sell_capacity > 0:
                orders.append(Order(product, ask_quote, -min(lot_size, sell_capacity)))

        # Too rich: sell more aggressively
        elif deviation > 0:
            bid_quote = int(round(fair_value - (base_half_spread + aggression) - inv_skew))
            ask_quote = int(round(fair_value - inv_skew))

            bid_quote, ask_quote = self.clamp_quotes(bid_quote, ask_quote, best_bid, best_ask)

            if sell_capacity > 0:
                orders.append(Order(product, ask_quote, -min(lot_size, sell_capacity)))
            if buy_capacity > 0:
                orders.append(Order(product, bid_quote, min(max(1, lot_size // 3), buy_capacity)))

        # Too cheap: buy more aggressively
        else:
            bid_quote = int(round(fair_value - inv_skew))
            ask_quote = int(round(fair_value + (base_half_spread + aggression) - inv_skew))

            bid_quote, ask_quote = self.clamp_quotes(bid_quote, ask_quote, best_bid, best_ask)

            if buy_capacity > 0:
                orders.append(Order(product, bid_quote, min(lot_size, buy_capacity)))
            if sell_capacity > 0:
                orders.append(Order(product, ask_quote, -min(max(1, lot_size // 3), sell_capacity)))

        return orders

    # =========================
    # Main run method
    # =========================
    def run(self, state: TradingState):
        POSITION_LIMITS = {
            "TOMATOES": 80,
            "EMERALDS": 80,
        }

        # Persistent state load
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

            mid_data = self.get_mid_price(order_depth)
            if mid_data is None:
                result[product] = orders
                continue

            _, _, mid_price = mid_data
            self.update_price_history(price_history, product, mid_price, max_len=60)

            # Strategy routing
            if product == "EMERALDS":
                orders = self.market_make_constant_mean(
                    product=product,
                    state=state,
                    order_depth=order_depth,
                    fair_value=10000,
                    pos_limit=POSITION_LIMITS[product],
                    half_spread=1,
                    lot_size=5,
                    inv_skew_per_unit=0,
                )

            elif product == "TOMATOES":
                orders = self.market_make_zscore(
                    product=product,
                    state=state,
                    order_depth=order_depth,
                    price_history=price_history,
                    pos_limit=POSITION_LIMITS[product],

                    mean_param=None,        # floating mean
                    mean_window=20,         # slower to move than 5
                    std_window=20,          # reacts to volatility spike faster
                    base_half_spread=2,     # wider quotes for noisier process
                    tight_z=0.8,            # only symmetric MM near fair
                    skew_z=2.0,             # only lean hard on true extremes
                    inv_skew_per_unit=0.02, # lighter inventory pressure
                    tight_size=3,
                    strong_size=5,
                    weak_buy_size=1,
                    weak_sell_size=1,
                    medium_buy_size=3,
                    medium_sell_size=3,
                )

            result[product] = orders

        traderData = json.dumps({
            "mid_history": price_history
        }, separators=(",", ":"))

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData