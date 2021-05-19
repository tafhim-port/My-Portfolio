from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import math
import re

import asyncio
import random

from typing import Optional

"""Constant listed from case packet"""
DAYS_IN_YEAR = 252
LAST_RATE_ROR_USD = 0.25
LAST_RATE_HAP_USD = 0.5
LAST_RATE_HAP_ROR = 2

TICK_SIZES = {'6RH': 0.00001, '6RM': 0.00001, '6RU': 0.00001, '6RZ': 0.00001, '6HH': 0.00002, \
    '6HM': 0.00002, '6HU': 0.00002, '6HZ': 0.00002, 'RHH': 0.0001, 'RHM': 0.0001, 'RHU': 0.0001, 'RHZ': 0.0001, "RORUSD": 0.00001}
FUTURES = [i+j for i in ["6R", "6H", "RH"] for j in ["H", "M", "U", "Z"]]
FUT_LIMIT = 80
#Caution Threshold above which we stop placing directional orders
CT = 30 

ROR_USD_FUTURES = [FUTURES[0], FUTURES[1], FUTURES[2], FUTURES[3]]
HAP_USD_FUTURES = [FUTURES[4], FUTURES[5], FUTURES[6], FUTURES[7]]
HAP_ROR_FUTURES = [FUTURES[8], FUTURES[9], FUTURES[10], FUTURES[11]]

FUTURE_EXP_ROR_USD = [FUTURES[1], FUTURES[2], FUTURES[3]]
FUTURE_EXP_HAP_USD = [FUTURES[5], FUTURES[6], FUTURES[7]]
FUTURE_EXP_HAP_ROR = [FUTURES[9], FUTURES[10], FUTURES[11]]

FIRSTEXP_FUTURES = [FUTURES[0], FUTURES[4], FUTURES[8]]
SECONDEXP_FUTURES = [FUTURES[1], FUTURES[5], FUTURES[9]]
THIRDEXP_FUTURES = [FUTURES[2], FUTURES[6], FUTURES[10]]
FOURTHEXP_FUTURES = [FUTURES[3], FUTURES[7], FUTURES[11]]

'''Rounds price to nearest tick_number above'''
def round_nearest(x, tick=0.0001):
    return round(round(x / tick) * tick, -int(math.floor(math.log10(tick))))

'''Finds daily interest rates from annual rate'''
def daily_rate(daily_rate):
    return math.pow(daily_rate, 1/252)

class PositionTrackerBot(UTCBot):
    """
    Bot that tracks its position, implements linear fading,
    and #prints out PnL information as
    computed by itself vs what was computed by the exchange
    """
    async def place_bids(self, asset):
        """
        Places and modifies a single bid, storing it by asset
        based upon the basic market making functionality
        """
        orders = await self.basic_mm(asset, self.fair[asset], self.params["edge"],
            self.params["size"], self.params["limit"],self.max_widths[asset])
        for index, price in enumerate(orders['bid_prices']):
            if self.pos[asset] >= CT:
                return
            if orders['bid_sizes'][index] != 0:
                resp = await self.modify_order(
                    self.bidorderid[asset][index],
                    asset,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    orders['bid_sizes'][index],
                    round_nearest(price, TICK_SIZES[asset]),
                )
                self.bidorderid[asset][index] = resp.order_id


    async def place_asks(self, asset):
        """
        Places and modifies a single bid, storing it by asset
        based upon the basic market making functionality
        """
        orders = await self.basic_mm(asset, self.fair[asset], self.params["edge"],
            self.params["size"], self.params["limit"],self.max_widths[asset])
        for index, price in enumerate(orders['ask_prices']):
            if self.pos[asset] <= -CT:
                return
            if orders['ask_sizes'][index] != 0:
                resp = await self.modify_order(
                    self.askorderid[asset][index],
                    asset,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    orders['ask_sizes'][index],
                    round_nearest(price, TICK_SIZES[asset]),
                )
                self.askorderid[asset][index] = resp.order_id
    
    async def spot_market(self):
        """
        Interaction within the spot market primarily consists
        of zeroing out the exposure to RORUSD exchange rates
        as best as possible, using market orders (assume spot
        market already is quite liquid)
        """
        net_position = self.pos["RORUSD"]
        for month in ["H", "M", "U", "Z"]:
            net_position += 0.05 * self.pos['RH' + month]
        net_position = round(net_position)
        print(f"{net_position} is net")
        bids_left = self.params["spot_limit"] - self.pos["RORUSD"]
        asks_left = self.params["spot_limit"] + self.pos["RORUSD"]

        print(f"bids_left = {bids_left}")
        print(f"asks_left = {asks_left}")


        if bids_left <= 0:
            resp = await self.place_order(
                "RORUSD",
                pb.OrderSpecType.MARKET,
                pb.OrderSpecSide.ASK,
                abs(bids_left)
            )
        elif asks_left <= 0:
            resp = await self.place_order(
                "RORUSD",
                pb.OrderSpecType.MARKET,
                pb.OrderSpecSide.BID,
                abs(asks_left)
            )
        elif (net_position > 0):
            resp = await self.place_order(
                "RORUSD",
                pb.OrderSpecType.MARKET,
                pb.OrderSpecSide.ASK,
                min(abs(net_position), asks_left)
            )
        elif (net_position < 0):
            resp = await self.place_order(
                "RORUSD",
                pb.OrderSpecType.MARKET,
                pb.OrderSpecSide.BID,
                min(abs(net_position), bids_left)
            )

    async def risk_reset(self):
        # Make sure we abide by risk limits on all the futures
        # By submitting MARKET orders

        unload_qty = 50

        requests = []

        for asset in FUTURES:
            if self.pos[asset] >= FUT_LIMIT:
                # we need to sell (market orders)
             
                resp = await self.place_order(
                    asset,
                    pb.OrderSpecType.MARKET,
                    pb.OrderSpecSide.BID,
                    unload_qty
                )

                print("Market selling")

            elif self.pos[asset] <= -FUT_LIMIT:
                # we need to buy (market orders)
                
                resp = await self.place_order(
                    asset,
                    pb.OrderSpecType.MARKET,
                    pb.OrderSpecSide.ASK,
                    unload_qty
                )
                
                print("Market buying")


    async def basic_mm(self, asset, fair, width, clip, max_pos, max_range):
        """
        Asset - Asset name on exchange
        Fair - Your prediction of the asset's true value
        Width - Your spread when quoting, i.e. difference between bid price and ask price
        Clip - Your maximum quote size on each level
        Max_Pos - The maximum number of contracts you are willing to hold (we just use risk limit here)
        Max_Range - The greatest you are willing to adjust your fair value by
        """

        ##The rate at which you fade is optimized so that you reach your max position
        ##at the same time you reach maximum range on the adjusted fair
        fade = (max_range / 2.0) / max_pos
        adjusted_fair = fair - self.pos[asset] * fade
        #adjusted_fair = fair
        ##Best bid, best ask prices
        bid_p = adjusted_fair - width / 2.0
        ask_p = adjusted_fair + width / 2.0

        ##Next best bid, ask price
        bid_p2 = min(adjusted_fair - clip * fade - width / 2.0,
            bid_p - TICK_SIZES[asset])
        ask_p2 = min(adjusted_fair + clip * fade + width / 2.0,
            ask_p + TICK_SIZES[asset])

        ##Remaining ability to quote
        bids_left = max_pos - self.pos[asset]
        asks_left = max_pos + self.pos[asset]


        if bids_left <= 0:
            #reduce your position as you are violating risk limits!
            ask_p = bid_p
            ask_s = clip
            #original:
            #ask_p2 = bid_p + TICK_SIZES[asset]
            ask_p2 = bid_p - TICK_SIZES[asset]
            ask_s2 = clip
            bid_s = 0
            bid_s2 = 0
        elif asks_left <= 0:
            #reduce your position as you are violating risk limits!
            bid_p = ask_p
            bid_s = clip
            #original
            #bid_p2 = ask_p - TICK_SIZES[asset]
            bid_p2 = ask_p + TICK_SIZES[asset]
            bid_s2 = clip
            ask_s = 0
            ask_s2 = 0
        else:
            #bid and ask size setting
            bid_s = min(bids_left, clip)
            bid_s2 = max(0, min(bids_left - clip, clip))
            ask_s = min(asks_left, clip)
            ask_s2 = max(0, min(asks_left - clip, clip))

        if self.pos[asset]<-CT:
            ask_p = 100
            ask_p2 = 100
            print(f"{asset} ask is 100")
            ask_s = 0
            ask_s2 = 0
        elif self.pos[asset]>CT:
            bid_p = 0.01
            bid_p2 = 0.01
            bid_s = 0
            bid_s2 = 0
            print(f"{asset} bid is 0.01")

        return {'asset': asset,
                'bid_prices': [bid_p, bid_p2],
                'bid_sizes': [bid_s, bid_s2],
                'ask_prices': [ask_p, ask_p2],
                'ask_sizes': [ask_s, ask_s2],
                'adjusted_fair': adjusted_fair,
                'fade': fade}

    async def handle_round_started(self):
        """
        Important variables below, some can be more dynamic to improve your case.
        Others are important to tracking pnl - cash, pos,
        Bidorderid, askorderid track order information so we can modify existing
        orders using the basic MM information (Right now only place 2 bids/2 asks max)
        """
        self.cash = 0.0
        self.pos = {asset:0 for asset in FUTURES + ["RORUSD"]}
        self.fair = {asset:5 for asset in FUTURES + ["RORUSD"]}
        self.mid = {asset: None for asset in FUTURES + ["RORUSD"]}


        self.historic_mid = {asset: None for asset in FUTURES + ["RORUSD"]}

        self.max_widths = {asset:0.002 for asset in FUTURES}

        self.bidorderid = {asset:["",""] for asset in FUTURES}
        self.askorderid = {asset:["",""] for asset in FUTURES}

        """
        Constant params with respect to assets. Modify this is you would like to change
        parameters based on asset
        """
        self.params = {
            #edge is spread
            "edge": 0.004, 
            "limit": 50,
            "size": 10,
            "spot_limit": 10
        }

        self.RATES_ROR_USD = [0.25]
        self.RATES_HAP_USD = [0.5]
        self.RATES_HAP_ROR = [2.0]
        self.t = 0

        self.ROR_interest_rates = []
        self.USD_interest_rates = []
        self.HAP_interest_rates = []

        self.ROR_federal_funds_rates = []
        self.HAP_federal_funds_rates = []
        self.USD_federal_funds_rates = []

        self.ROR_DAILY = 0.0
        self.HAP_DAILY = 0.0
        self.USD_DAILY = 0.0


        for asset in FUTURES:
            resp = await self.place_order(
                asset,
                pb.OrderSpecType.LIMIT,
                pb.OrderSpecSide.ASK,
                100,
                3000
            )

            resp2 = await self.modify_order(
                asset,
                pb.OrderSpecType.LIMIT,
                pb.OrderSpecSide.BID,
                100,
                0.0005
            )




    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        #Possible exchange updates: 'market_snapshot_msg','fill_msg'
        #'liquidation_msg','generic_msg', 'trade_msg', 'pnl_msg', etc.
        """
        Calculate PnL based upon market to market contracts and tracked cash
        """
        if kind == "pnl_msg":
            #print(update.realized_pnl)
            #print(update.m2m_pnl)
            my_m2m = self.cash
            for asset in (FUTURES + ["RORUSD"]):
               my_m2m += self.mid[asset] * self.pos[asset] if self.mid[asset] is not None else 0
            print(my_m2m)
            ##print("M2M", update.pnl_msg.m2m_pnl, my_m2m)
        #Update position upon fill messages of your trades
        elif kind == "fill_msg":
            print(f"{update.fill_msg.asset} = ,  {update.fill_msg.filled_qty}")
            if update.fill_msg.order_side == pb.FillMessageSide.BUY:
                self.cash -= update.fill_msg.filled_qty * float(update.fill_msg.price)
                self.pos[update.fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.cash += update.fill_msg.filled_qty * float(update.fill_msg.price)
                self.pos[update.fill_msg.asset] -= update.fill_msg.filled_qty
            print(self.pos)
            #await self.spot_market()
            #for asset in FUTURES:
                #await self.place_bids(asset)
                #await self.place_asks(asset)
            #await self.spot_market()
            #await self.risk_reset()
        #Identify mid price through order book updates
        elif kind == "market_snapshot_msg":
            for asset in (FUTURES + ["RORUSD"]):
                book = update.market_snapshot_msg.books[asset]

                mid: "Optional[float]"
                if len(book.asks) > 0:
                    if len(book.bids) > 0:
                        mid = (float(book.asks[0].px) + float(book.bids[0].px)) / 2
                        self.historic_mid[asset] = mid
                    else:
                        if self.historic_mid[asset] != None:
                            mid = self.historic_mid[asset]
                        else:
                            mid = float(book.asks[0].px)
                elif len(book.bids) > 0:
                    if self.historic_mid[asset] != None:
                        mid = self.historic_mid[asset]
                    else:
                        mid = float(book.bids[0].px)
                else:
                    # set mid to last known mid price (if it exists)
                    if self.historic_mid[asset] != None:
                        mid = self.historic_mid[asset]
                    else:
                        mid = self.fair[asset]

                self.mid[asset] = mid
        ##print(self.mid[asset])




        ###########



        ############

        #Updating Messages for Fed Funds Rate and Daily Interest Rates
        elif kind == "generic_msg":

            split_string = update.generic_msg.message.split()
            fed_update = False
            ROR_update = False
            HAP_update = False
            USD_update = False
            threshold = 0.005

            if (split_string[0] == 'ROR'):
                self.ROR_federal_funds_rates.append(float(split_string[-1]))

                fed_update = True
                ROR_update = True
            elif (split_string[0] == 'HAP'):
                self.HAP_federal_funds_rates.append(float(split_string[-1]))
                fed_update = True
                HAP_update = True
            elif (split_string[0] == 'USD'):
                fed_update = True
                USD_update = True
                self.USD_federal_funds_rates.append(float(split_string[-1]))

            #get the tick time mark through "['201,', '1.0331198021823988,', '1.041816986957577,', '1.063806863247064']" which also gives us interest rates
            elif re.sub("[^0-9]", "", split_string[0]) != "":
                self.t = int(float((split_string[0].replace(',', ''))))

                self.ROR_interest_rates = (float((split_string[1].replace(',', ''))))
                self.HAP_interest_rates = (float((split_string[2].replace(',', ''))))
                self.USD_interest_rates = (float((split_string[3].replace(',', ''))))


                self.ROR_DAILY = math.pow(self.ROR_interest_rates, 1/252)
                self.HAP_DAILY = math.pow(self.HAP_interest_rates, 1/252)
                self.USD_DAILY = math.pow(self.USD_interest_rates, 1/252)


            if fed_update and self.t>0:
                print("EXECUTING SHIFT")
                #cancel all existing orders from the book
                # Cancel all orders

                cancel_requests = []
                for asset in FUTURES:
                    for index in range(0, 2):
                        cancel_requests.append(self.cancel_order(self.bidorderid[asset][index]))
                        cancel_requests.append(self.cancel_order(self.askorderid[asset][index]))

                cancel_responses = await asyncio.gather(*cancel_requests)

            
                order_requests = []

    
                # max parameter
                max_size = 30
                #this section will be updating the "shift" in interest rates caused by an update from the exchange giving new federal funds rates
                #we are trying to capture this shift before competitors 
                if USD_update:
                    if(self.USD_federal_funds_rates[-1] - self.USD_interest_rates>threshold):
                        #positive side USD
                        #ROR/USD market buy
                        for asset_name in ROR_USD_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass

                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK,min(max_size,CT-self.pos[asset_name])))
                        #HAP/USD market buy
                        for asset_name in HAP_USD_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK,min(max_size,CT-self.pos[asset_name])))
                        pass

                    elif(self.USD_federal_funds_rates[-1] - self.USD_interest_rates<-threshold):
                        #negative side USD
                        #ROR/USD market sell
                        for asset_name in ROR_USD_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID,min(max_size,abs(CT-self.pos[asset_name]))))
                        #HAP/USD market sell
                        for asset_name in HAP_USD_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID,min(max_size,abs(CT-self.pos[asset_name]))))
                        pass

                elif ROR_update:
                    if (self.ROR_federal_funds_rates[-1] - self.ROR_interest_rates > threshold):
                        # positive side ROR
                        #ROR/USD market sell
                        for asset_name in ROR_USD_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID,min(max_size,abs(CT-self.pos[asset_name]))))
                        #HAP/ROR market buy
                        for asset_name in HAP_ROR_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK,min(max_size,CT-self.pos[asset_name])))

                    elif (self.ROR_federal_funds_rates[-1] - self.ROR_interest_rates < -threshold):
                        # negative side ROR
                        #ROR/USD market buy
                        for asset_name in ROR_USD_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK,min(max_size,CT-self.pos[asset_name])))
                        #HAP/ROR market sell
                        for asset_name in HAP_ROR_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID,min(max_size,abs(CT-self.pos[asset_name]))))

                elif HAP_update:
                    if (self.HAP_federal_funds_rates[-1] - self.HAP_interest_rates > threshold):
                        # positive side hap
                        #HAP/USD market sell
                        for asset_name in HAP_USD_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID,min(max_size,abs(CT-self.pos[asset_name]))))
                        #HAP/ror market sell
                        for asset_name in HAP_ROR_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID,min(max_size,abs(CT-self.pos[asset_name]))))

                    elif (self.HAP_federal_funds_rates[-1] - self.HAP_interest_rates < -threshold):
                        # negative side hap
                        #HAP/USD market buy
                        for asset_name in HAP_USD_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK,min(max_size,CT-self.pos[asset_name])))
                        #HAP/ror market buy
                        for asset_name in HAP_ROR_FUTURES:
                            if asset_name[-1] =="H" and self.t>63:
                                pass
                            if asset_name[-1] =="M" and self.t>126:
                                pass
                            if asset_name[-1] =="U" and self.t>189:
                                pass
                            order_requests.append(self.place_order(asset_name, pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK,min(max_size,CT-self.pos[asset_name])))

                if len(order_requests) > 0:
                    order_responses = await asyncio.gather(*order_requests)



            #self.ROR_DAILY = ROR_DAILY
            #self.HAP_DAILY = HAP_DAILY
            #self.USD_DAILY = USD_DAILY

            mod = 2

            if not fed_update and self.t % mod == 0:
                await self.evaluate_fairs()
                # await self.spot_market()

                for asset in FUTURES:
                    await self.place_bids(asset)
                    await self.place_asks(asset)
                await self.spot_market()
                await self.risk_reset()


    async def evaluate_fairs(self):
        #Last Exchange rates given on December 31,2009 (Must update this part from competition)

        #pricing front month contracts
        self.RATES_ROR_USD.append(self.RATES_ROR_USD[-1] * ((1+self.ROR_DAILY)/(1+self.USD_DAILY)))
        self.fair[FUTURES[0]] = self.RATES_ROR_USD[-1]
        self.RATES_HAP_USD.append(self.RATES_HAP_USD[-1] * ((1+self.HAP_DAILY)/(1+self.USD_DAILY)))
        self.fair[FUTURES[4]] = self.RATES_HAP_USD[-1]
        self.RATES_HAP_ROR.append(self.RATES_HAP_ROR[-1] * ((1+self.HAP_DAILY)/(1+self.ROR_DAILY)))
        self.fair[FUTURES[8]] = self.RATES_HAP_ROR[-1]

        #pricing backend futures contracts
        for asset in FUTURE_EXP_ROR_USD:
            self.fair[asset] = self.RATES_ROR_USD[-1] * ((self.USD_DAILY*((252-self.t)/252)+1)/(self.ROR_DAILY*((252-self.t)/252)+1))
        for asset in FUTURE_EXP_HAP_USD:
            self.fair[asset] = self.RATES_HAP_USD[-1] * ((self.USD_DAILY*((252-self.t)/252)+1)/(self.HAP_DAILY*((252-self.t)/252)+1))
        for asset in FUTURE_EXP_HAP_ROR:
            self.fair[asset] = self.RATES_HAP_ROR[-1] * ((self.ROR_DAILY*((252-self.t)/252)+1)/(self.HAP_DAILY*((252-self.t)/252)+1))


if __name__ == "__main__":
    start_bot(PositionTrackerBot)
