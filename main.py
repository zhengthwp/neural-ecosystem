#!/usr/bin/env python3
"""
SET100 Daily Price Tracker
===========================
Tracks the SET100 Index (Thailand) daily opening price.
Data is fetched from Yahoo Finance (SET100.BK) once per day and stored locally.

Commands
--------
  update              Fetch today's SET100 snapshot and store it
  show  [--days N]    Show last N days of stored prices  (default 30)
  show  --date DATE   Show a specific date
  today               Show today's price
  enter               Manually enter/overwrite a day's price
  import-csv FILE     Bulk-import from a CSV file (date,open,high,low,close)

CSV format example:
  date,open,high,low,close
  2026-04-01,2200.50,2250.00,2190.00,2230.00
"""

import argparse
import csv
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.prompt import FloatPrompt, Prompt

import db
import fetcher

console = Console()

LOG_PATH = Path(__file__).parent / "set100_tracker.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─── commands ────────────────────────────────────────────────────────────────

def cmd_update(args):
    db.init_db()
    logging.info("update started")
    console.print("[cyan]Fetching today's SET100 snapshot from Yahoo Finance…[/cyan]")
    record = fetcher.fetch_today()

    if not record:
        msg = "No data returned — market may be closed, holiday, or weekend"
        logging.warning(msg)
        console.print(
            f"[yellow]{msg}.\n"
            "Use [bold]python main.py enter[/bold] to input prices manually.[/yellow]"
        )
        return

    record["source"] = "yahoo"
    db.upsert_prices([record])
    logging.info(
        "stored date=%s open=%.2f high=%.2f low=%.2f close=%.2f",
        record["date"], record["open"], record["high"], record["low"], record["close"],
    )
    console.print(f"[green]Stored SET100 data for {record['date']}.[/green]")
    _print_single(record)


def cmd_show(args):
    db.init_db()
    total = db.count()

    if args.date:
        record = db.fetch_price_by_date(args.date)
        if not record:
            console.print(f"[yellow]No data for {args.date}. Available records: {total}[/yellow]")
            return
        _print_table([record], title=f"SET100 — {args.date}")
    else:
        days = args.days or 30
        records = db.fetch_prices(limit=days)
        if not records:
            console.print(
                "[yellow]No data in database yet.\n"
                "Run: [bold]python main.py update[/bold]  to fetch today's price.[/yellow]"
            )
            return
        _print_table(records, title=f"SET100 Index — Last {len(records)} trading days")


def cmd_today(args):
    db.init_db()
    today_str = str(date.today())
    record = db.fetch_price_by_date(today_str)

    if record:
        _print_single(record)
        return

    console.print("[cyan]Today's data not in DB — fetching from Yahoo Finance…[/cyan]")
    record = fetcher.fetch_today()
    if record:
        record["source"] = "yahoo"
        db.upsert_prices([record])
        _print_single(record)
    else:
        console.print(
            "[yellow]Could not fetch today's data.\n"
            "The market may not have opened yet, or it's a holiday/weekend.\n"
            "Use [bold]python main.py enter[/bold] to input prices manually.[/yellow]"
        )


def cmd_enter(args):
    """Manually enter / overwrite a day's price."""
    db.init_db()
    console.print("[bold]Manual price entry for SET100[/bold]\n")

    if args.date:
        date_str = args.date
    else:
        date_str = Prompt.ask("Date (YYYY-MM-DD)", default=str(date.today()))

    open_p  = FloatPrompt.ask("  Open")
    high    = FloatPrompt.ask("  High", default=open_p)
    low     = FloatPrompt.ask("  Low",  default=open_p)
    close   = FloatPrompt.ask("  Close")
    prev_cl = FloatPrompt.ask("  Prev Close (optional, press Enter to skip)", default=0.0)

    record = {
        "date":       date_str,
        "open":       round(open_p, 2),
        "high":       round(max(high, open_p, close), 2),
        "low":        round(min(low,  open_p, close), 2),
        "close":      round(close, 2),
        "prev_close": round(prev_cl, 2) if prev_cl else None,
        "volume":     0,
        "source":     "manual",
    }
    db.upsert_prices([record])
    console.print(f"\n[green]Saved SET100 entry for {date_str}.[/green]")
    _print_single(record)


def cmd_export_csv(args):
    """Export all stored prices to data/set100_prices.csv."""
    db.init_db()
    records = db.fetch_prices(limit=99999)  # all records
    records.sort(key=lambda r: r["date"])   # ascending

    out = Path(__file__).parent / "data" / "set100_prices.csv"
    out.parent.mkdir(exist_ok=True)

    fieldnames = ["date", "open", "high", "low", "close", "prev_close", "volume", "source"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    console.print(f"[green]Exported {len(records)} record(s) to {out}[/green]")


def cmd_import_csv(args):
    """Bulk import from a CSV file."""
    db.init_db()
    path = args.file
    records = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "date":       row["date"].strip(),
                    "open":       float(row["open"]),
                    "high":       float(row.get("high", row["open"])),
                    "low":        float(row.get("low",  row["open"])),
                    "close":      float(row.get("close", row["open"])),
                    "prev_close": float(row["prev_close"]) if row.get("prev_close") else None,
                    "volume":     int(float(row.get("volume", 0))),
                    "source":     "csv",
                })
    except FileNotFoundError:
        console.print(f"[red]File not found: {path}[/red]")
        return
    except (KeyError, ValueError) as e:
        console.print(f"[red]CSV parse error: {e}[/red]")
        return

    db.upsert_prices(records)
    console.print(f"[green]Imported {len(records)} record(s) from {path}.[/green]")
    _print_table(records[:10], title="First 10 imported rows")


def cmd_status(args):
    """Health-check: show last fetch, streak, and any gaps."""
    db.init_db()
    total   = db.count()
    records = db.fetch_prices(limit=30)
    today   = str(date.today())

    console.print()
    console.rule("[bold]SET100 Tracker — Status[/bold]")

    # ── database summary ──
    latest = db.latest_date()
    console.print(f"  [bold]Total records in DB  :[/bold] {total}")
    console.print(f"  [bold]Most recent date     :[/bold] {latest or 'none'}")
    console.print(f"  [bold]Today                :[/bold] {today}")

    # ── today's data present? ──
    has_today = db.fetch_price_by_date(today) is not None
    if has_today:
        console.print("  [bold]Today's data         :[/bold] [green]YES — already fetched[/green]")
    else:
        console.print("  [bold]Today's data         :[/bold] [yellow]NOT YET — run: python main.py update[/yellow]")

    # ── streak (consecutive days with data, going back from latest) ──
    if records:
        streak = 0
        check  = date.fromisoformat(records[0]["date"])
        dates  = {r["date"] for r in records}
        while str(check) in dates:
            streak += 1
            check  -= timedelta(days=1)
        console.print(f"  [bold]Consecutive days     :[/bold] {streak} (counting back, incl. weekends)")

    # ── gaps in last 30 records ──
    if len(records) >= 2:
        gaps = []
        for i in range(len(records) - 1):
            d1 = date.fromisoformat(records[i]["date"])
            d2 = date.fromisoformat(records[i + 1]["date"])
            delta = (d1 - d2).days
            if delta > 3:  # more than a long weekend
                gaps.append((records[i + 1]["date"], records[i]["date"], delta))
        if gaps:
            console.print(f"  [bold]Gaps (>{3} days)       :[/bold] [red]{len(gaps)} gap(s) found[/red]")
            for g in gaps[:5]:
                console.print(f"    • {g[0]} → {g[1]}  ({g[2]} days)")
        else:
            console.print("  [bold]Gaps                 :[/bold] [green]None in last 30 records[/green]")

    # ── log file ──
    console.print(f"\n  [bold]Log file             :[/bold] {LOG_PATH}")
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text().splitlines()
        recent = lines[-5:] if len(lines) >= 5 else lines
        console.print("  [bold]Last 5 log lines     :[/bold]")
        for ln in recent:
            console.print(f"    [dim]{ln}[/dim]")
    else:
        console.print("  [dim](no log file yet — run update first)[/dim]")

    console.print()


# ─── display helpers ──────────────────────────────────────────────────────────

def _print_single(r: dict):
    open_p  = r.get("open") or 0
    close_p = r.get("close") or 0
    prev_cl = r.get("prev_close")
    base    = prev_cl if prev_cl else open_p
    change  = close_p - base
    pct     = (change / base * 100) if base else 0
    arrow   = "▲" if change >= 0 else "▼"
    color   = "green" if change >= 0 else "red"
    source  = r.get("source", "yahoo")

    console.print()
    console.rule(f"[bold]SET100 Index — {r['date']}[/bold]")
    console.print(f"  [bold]Open      :[/bold] [yellow]{open_p:,.2f}[/yellow]")
    console.print(f"  [bold]High      :[/bold] {r.get('high', 0):,.2f}")
    console.print(f"  [bold]Low       :[/bold] {r.get('low', 0):,.2f}")
    console.print(f"  [bold]Close     :[/bold] {close_p:,.2f}  [{color}]{arrow} {abs(change):,.2f} ({pct:+.2f}%)[/{color}]")
    if prev_cl:
        console.print(f"  [bold]Prev Close:[/bold] {prev_cl:,.2f}")
    console.print(f"  [bold]Source    :[/bold] [dim]{source}[/dim]")
    console.print()


def _print_table(records: list, title: str = "SET100 Daily Prices"):
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Date",       style="cyan",   justify="center", min_width=12)
    table.add_column("Open",       style="yellow", justify="right",  min_width=10)
    table.add_column("High",                       justify="right",  min_width=10)
    table.add_column("Low",                        justify="right",  min_width=10)
    table.add_column("Close",                      justify="right",  min_width=10)
    table.add_column("Chg",                        justify="right",  min_width=9)
    table.add_column("Chg %",                      justify="right",  min_width=8)
    table.add_column("Src",                        justify="center", min_width=6)

    for r in records:
        open_p  = r.get("open")  or 0
        close_p = r.get("close") or 0
        prev_cl = r.get("prev_close")
        base    = prev_cl if prev_cl else open_p
        change  = close_p - base
        pct     = (change / base * 100) if base else 0
        color   = "green" if change >= 0 else "red"
        arrow   = "▲" if change >= 0 else "▼"
        src     = (r.get("source") or "yahoo")[0].upper()  # Y/M/C

        table.add_row(
            r["date"],
            f"{open_p:,.2f}",
            f"{r.get('high', 0):,.2f}",
            f"{r.get('low', 0):,.2f}",
            f"{close_p:,.2f}",
            Text(f"{arrow} {abs(change):,.2f}", style=color),
            Text(f"{pct:+.2f}%",               style=color),
            src,
        )

    console.print()
    console.print(table)
    console.print(
        "  [dim]Src: Y=Yahoo Finance · M=Manual · C=CSV import[/dim]\n"
        "  [dim]Chg: vs Prev Close when available, else vs Open[/dim]\n"
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SET100 Daily Index Price Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("update",     help="Fetch and store today's SET100 price")
    sub.add_parser("today",      help="Show today's SET100 opening price")
    sub.add_parser("status",     help="Health-check: last fetch, gaps, log tail")
    sub.add_parser("export-csv", help="Export all data to data/set100_prices.csv")

    shw = sub.add_parser("show", help="Display stored price history")
    shw.add_argument("--days", type=int, default=30, metavar="N")
    shw.add_argument("--date", metavar="YYYY-MM-DD")

    ent = sub.add_parser("enter", help="Manually enter a day's price")
    ent.add_argument("--date", metavar="YYYY-MM-DD")

    imp = sub.add_parser("import-csv", help="Bulk import from CSV file")
    imp.add_argument("file", metavar="FILE")

    args = parser.parse_args()

    dispatch = {
        "update":     cmd_update,
        "show":       cmd_show,
        "today":      cmd_today,
        "enter":      cmd_enter,
        "import-csv": cmd_import_csv,
        "export-csv": cmd_export_csv,
        "status":     cmd_status,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
