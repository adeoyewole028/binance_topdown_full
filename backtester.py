"""Explicit UTC historical research CLI; offline input is the default."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from configuration import validated_config
from historical_data import fetch_history, requirements, validate_history
from historical_replay import replay
from paper_policy import validated_policy, number
from strategy_profiles import PROFILES


def utc_milliseconds(value: str) -> int:
    date=datetime.fromisoformat(value.replace('Z','+00:00'))
    if len(value)==10:date=date.replace(tzinfo=timezone.utc)
    if date.tzinfo is None or date.utcoffset()!=timezone.utc.utcoffset(date):
        raise ValueError('UTC dates or explicitly UTC timestamps required')
    return int(date.timestamp()*1000)


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path,help='Captured dataset JSON with explicit as_of')
    source.add_argument('--fetch',action='store_true',help='Explicit public requests using existing shared host state')
    for name in ('config','policy','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('symbol','start','end','quantity-step'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--profile',choices=list(PROFILES),required=True)
    parser.add_argument('--venue',choices=['testnet','mainnet'],required=True)
    parser.add_argument('--sizing',choices=['fixed','risk'],required=True)
    parser.add_argument('--config-version',type=int,required=True)
    parser.add_argument('--state-directory',type=Path)
    parser.add_argument('--account')
    parser.add_argument('--state-mode',choices=['paper','exchange'])
    parser.add_argument('--save-data',type=Path)
    args=parser.parse_args(argv)
    config=validated_config(json.loads(args.config.read_text(encoding='utf-8')))
    policy=validated_policy(json.loads(args.policy.read_text(encoding='utf-8')))
    start,end=utc_milliseconds(args.start),utc_milliseconds(args.end)
    requirements(args.profile,start,end)
    validate_history({},symbol=args.symbol,venue=args.venue,profile=args.profile,start=start,end=end,as_of=end)
    if (config['strategy_profile']!=args.profile or args.symbol.split('/')[1]!=policy['quote']
            or args.config_version<1 or number(args.quantity_step)<=0):
        raise ValueError('Consistent explicit profile, quote, version and precision required')
    paths=[args.config,args.policy,*([args.data] if args.data else []),args.output,
           *([args.save_data] if args.save_data else [])]
    if len({path.resolve() for path in paths})!=len(paths):
        raise ValueError('Input/output paths must be distinct')
    config=validated_config({**config,'paper_policy':policy})
    if args.fetch:
        if args.state_directory is None or not args.account or not args.state_mode:
            parser.error('--fetch requires existing --state-directory, --account and --state-mode')
        from state_store import StateStore
        from request_budget import RequestBudget
        from runtime_environment import RuntimeEnvironment, construct_exchange
        from market_adapter import BinancePublicMarket
        from historical_source import BudgetedHistory
        store=StateStore(args.state_directory.resolve(),args.account,args.venue,args.state_mode)
        store.validate()
        runtime=RuntimeEnvironment(False,'paper',args.venue,False,False,'','',args.state_directory.resolve())
        adapter=BudgetedHistory(BinancePublicMarket(construct_exchange(runtime)),RequestBudget(store))
        as_of=adapter.capture_time(args.symbol)
        dataset=fetch_history(adapter,symbol=args.symbol,venue=args.venue,profile=args.profile,
            start=start,end=end,as_of=as_of)
    else:
        dataset=json.loads(args.data.read_text(encoding='utf-8'))
        if any(dataset.get(key)!=value for key,value in {'symbol':args.symbol,'venue':args.venue,
                'profile':args.profile,'start':start,'end':end}.items()):
            raise ValueError('Captured dataset does not match explicit study inputs')
    result=replay(dataset,config=config,policy=policy,sizing=args.sizing,quantity_step=args.quantity_step,
                  namespace=f'research:{args.venue}:offline',config_version=args.config_version)
    if args.save_data:
        args.save_data.write_text(json.dumps(dataset,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return 0 if result['state']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
