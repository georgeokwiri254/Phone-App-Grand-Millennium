"""
Grand Millennium Revenue Analytics - AED Currency Handler

Comprehensive AED (Arab Emirates Dirham) currency handling for the mobile game.
Handles formatting, conversions, calculations, and mobile display optimization.
"""

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Dict, List, Optional
import locale
from datetime import datetime

class AEDCurrencyHandler:
    """Complete AED currency handling for mobile game"""
    
    # AED currency constants
    AED_SYMBOL = "د.إ"  # Arabic AED symbol
    AED_SYMBOL_EN = "AED"  # English abbreviation
    AED_CODE = "784"  # ISO currency code
    
    # Formatting options
    DECIMAL_PLACES = 2
    THOUSANDS_SEPARATOR = ","
    DECIMAL_SEPARATOR = "."
    
    def __init__(self, locale_preference="ar_AE"):
        """Initialize AED handler with locale preference"""
        self.locale_preference = locale_preference
        self.setup_locale()
        
        # Exchange rates (if needed for future multi-currency support)
        self.exchange_rates = {
            "USD": 3.67,  # 1 USD = 3.67 AED (approximate)
            "EUR": 4.02,  # 1 EUR = 4.02 AED (approximate)
            "GBP": 4.65,  # 1 GBP = 4.65 AED (approximate)
        }
    
    def setup_locale(self):
        """Setup locale for proper number formatting"""
        try:
            if self.locale_preference == "ar_AE":
                # Arabic UAE locale
                locale.setlocale(locale.LC_ALL, 'ar_AE.UTF-8')
            elif self.locale_preference == "en_AE":
                # English UAE locale
                locale.setlocale(locale.LC_ALL, 'en_AE.UTF-8')
            else:
                # Fallback to system default
                locale.setlocale(locale.LC_ALL, '')
        except locale.Error:
            # Fallback if locale not available
            pass
    
    def format_aed(self, 
                   amount: Union[int, float, Decimal], 
                   include_symbol: bool = True,
                   decimal_places: Optional[int] = None,
                   compact: bool = False) -> str:
        """
        Format amount as AED currency with proper Arabic formatting
        
        Args:
            amount: Amount to format
            include_symbol: Include AED symbol
            decimal_places: Override default decimal places
            compact: Use compact notation (K, M, B)
        """
        if amount is None:
            return f"{self.AED_SYMBOL} 0" if include_symbol else "0"
        
        # Convert to Decimal for precise calculations
        if isinstance(amount, (int, float)):
            amount = Decimal(str(amount))
        
        # Round to specified decimal places
        places = decimal_places if decimal_places is not None else self.DECIMAL_PLACES
        amount = amount.quantize(Decimal('0.01' if places == 2 else f'0.{"0" * places}'), 
                                rounding=ROUND_HALF_UP)
        
        # Handle compact notation for large amounts
        if compact and abs(amount) >= 1000:
            return self._format_compact_aed(amount, include_symbol)
        
        # Format with thousands separators
        if places == 0:
            formatted = f"{amount:,.0f}"
        else:
            formatted = f"{amount:,.{places}f}"
        
        # Add AED symbol
        if include_symbol:
            return f"{self.AED_SYMBOL} {formatted}"
        else:
            return formatted
    
    def _format_compact_aed(self, amount: Decimal, include_symbol: bool = True) -> str:
        """Format large amounts with compact notation (K, M, B)"""
        abs_amount = abs(amount)
        sign = "-" if amount < 0 else ""
        
        if abs_amount >= 1_000_000_000:
            # Billions
            value = abs_amount / 1_000_000_000
            suffix = "B"
        elif abs_amount >= 1_000_000:
            # Millions
            value = abs_amount / 1_000_000
            suffix = "M"
        elif abs_amount >= 1_000:
            # Thousands
            value = abs_amount / 1_000
            suffix = "K"
        else:
            return self.format_aed(amount, include_symbol, compact=False)
        
        # Format with 1 decimal place for compact notation
        if value >= 100:
            formatted = f"{sign}{value:.0f}{suffix}"
        else:
            formatted = f"{sign}{value:.1f}{suffix}"
        
        if include_symbol:
            return f"{self.AED_SYMBOL} {formatted}"
        else:
            return formatted
    
    def parse_aed_string(self, aed_string: str) -> Optional[Decimal]:
        """Parse AED string back to Decimal amount"""
        if not aed_string:
            return None
        
        # Remove AED symbols and whitespace
        cleaned = aed_string.replace(self.AED_SYMBOL, "").replace(self.AED_SYMBOL_EN, "")
        cleaned = cleaned.strip()
        
        # Handle compact notation
        if cleaned.endswith('K'):
            multiplier = 1_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith('M'):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith('B'):
            multiplier = 1_000_000_000
            cleaned = cleaned[:-1]
        else:
            multiplier = 1
        
        # Remove thousands separators
        cleaned = cleaned.replace(self.THOUSANDS_SEPARATOR, "")
        
        try:
            amount = Decimal(cleaned) * multiplier
            return amount
        except (ValueError, TypeError):
            return None
    
    def calculate_percentage_change(self, old_amount: Union[int, float, Decimal], 
                                   new_amount: Union[int, float, Decimal]) -> Dict[str, Union[Decimal, str]]:
        """Calculate percentage change between two AED amounts"""
        if old_amount == 0:
            return {
                'change_amount': Decimal(str(new_amount)),
                'change_percentage': Decimal('0'),
                'formatted_change': f"+{self.format_aed(new_amount)}",
                'formatted_percentage': "+∞%"
            }
        
        old_decimal = Decimal(str(old_amount))
        new_decimal = Decimal(str(new_amount))
        
        change_amount = new_decimal - old_decimal
        change_percentage = (change_amount / old_decimal) * 100
        
        # Format changes with proper signs
        change_sign = "+" if change_amount >= 0 else ""
        percentage_sign = "+" if change_percentage >= 0 else ""
        
        return {
            'change_amount': change_amount,
            'change_percentage': change_percentage,
            'formatted_change': f"{change_sign}{self.format_aed(change_amount)}",
            'formatted_percentage': f"{percentage_sign}{change_percentage:.1f}%"
        }
    
    def format_mobile_display(self, amount: Union[int, float, Decimal], 
                             context: str = "general") -> str:
        """Format AED for mobile display with context-specific optimization"""
        contexts = {
            "dashboard_card": {"compact": True, "decimal_places": 0},
            "challenge_reward": {"compact": False, "decimal_places": 0},
            "pricing_slider": {"compact": False, "decimal_places": 0},
            "segment_budget": {"compact": True, "decimal_places": 0},
            "leaderboard": {"compact": True, "decimal_places": 0},
            "achievement": {"compact": False, "decimal_places": 0},
            "detailed_report": {"compact": False, "decimal_places": 2}
        }
        
        options = contexts.get(context, {"compact": False, "decimal_places": 0})
        return self.format_aed(amount, **options)
    
    def get_color_for_amount(self, amount: Union[int, float, Decimal], 
                            threshold_positive: Union[int, float] = 0) -> tuple:
        """Get color tuple for amount based on positive/negative value"""
        if amount is None:
            return (0.5, 0.5, 0.5, 1)  # Gray for null
        
        if amount > threshold_positive:
            return (0.0, 0.8, 0.0, 1)  # Green for positive
        elif amount < 0:
            return (0.8, 0.0, 0.0, 1)  # Red for negative
        else:
            return (1.0, 0.84, 0.0, 1)  # Gold for neutral/zero
    
    def format_revenue_target(self, current: Union[int, float, Decimal], 
                             target: Union[int, float, Decimal]) -> Dict[str, str]:
        """Format revenue target with progress indicators"""
        current_decimal = Decimal(str(current))
        target_decimal = Decimal(str(target))
        
        progress_percentage = min((current_decimal / target_decimal) * 100, 100) if target_decimal > 0 else 0
        remaining = max(target_decimal - current_decimal, 0)
        
        return {
            'current': self.format_aed(current_decimal, compact=True),
            'target': self.format_aed(target_decimal, compact=True),
            'remaining': self.format_aed(remaining, compact=True),
            'progress_percentage': f"{progress_percentage:.1f}%",
            'status': "achieved" if current_decimal >= target_decimal else "in_progress"
        }
    
    def calculate_game_score_from_aed(self, aed_amount: Union[int, float, Decimal], 
                                     multiplier: float = 1.0) -> int:
        """Convert AED amount to game score points"""
        if aed_amount is None:
            return 0
        
        amount_decimal = Decimal(str(aed_amount))
        # Base conversion: 1 AED = 1 point, then apply multiplier
        base_score = int(amount_decimal * Decimal(str(multiplier)))
        
        return max(0, base_score)  # Ensure non-negative score
    
    def format_achievement_reward(self, aed_amount: Union[int, float, Decimal]) -> str:
        """Format AED amount for achievement rewards"""
        return f"🏆 Reward: {self.format_aed(aed_amount, compact=True)}"
    
    def get_budget_allocation_display(self, allocations: Dict[str, float], 
                                     total_budget: Union[int, float, Decimal]) -> List[Dict]:
        """Format budget allocation for segment game display"""
        total_budget_decimal = Decimal(str(total_budget))
        formatted_allocations = []
        
        for segment, percentage in allocations.items():
            amount = (total_budget_decimal * Decimal(str(percentage))) / 100
            formatted_allocations.append({
                'segment': segment,
                'percentage': f"{percentage:.1f}%",
                'amount': self.format_aed(amount, compact=True),
                'amount_raw': amount
            })
        
        return formatted_allocations
    
    def validate_aed_input(self, input_string: str) -> Dict[str, Union[bool, str, Decimal]]:
        """Validate user input for AED amounts"""
        result = {
            'valid': False,
            'amount': None,
            'error_message': None,
            'formatted': None
        }
        
        if not input_string or not input_string.strip():
            result['error_message'] = "Amount cannot be empty"
            return result
        
        # Try to parse the input
        parsed_amount = self.parse_aed_string(input_string)
        
        if parsed_amount is None:
            result['error_message'] = "Invalid amount format"
            return result
        
        if parsed_amount < 0:
            result['error_message'] = "Amount cannot be negative"
            return result
        
        if parsed_amount > Decimal('999999999'):
            result['error_message'] = "Amount too large"
            return result
        
        result['valid'] = True
        result['amount'] = parsed_amount
        result['formatted'] = self.format_aed(parsed_amount)
        
        return result

class MobileAEDFormatter:
    """Specialized AED formatter for mobile game interface"""
    
    def __init__(self):
        self.currency_handler = AEDCurrencyHandler()
    
    def format_dashboard_kpi(self, amount: Union[int, float, Decimal], kpi_type: str) -> str:
        """Format KPI values for dashboard cards"""
        context_map = {
            'revenue': 'dashboard_card',
            'adr': 'dashboard_card', 
            'revpar': 'dashboard_card',
            'forecast': 'dashboard_card'
        }
        
        context = context_map.get(kpi_type, 'dashboard_card')
        return self.currency_handler.format_mobile_display(amount, context)
    
    def format_challenge_points(self, aed_amount: Union[int, float, Decimal]) -> str:
        """Format AED amount as challenge points"""
        points = self.currency_handler.calculate_game_score_from_aed(aed_amount)
        return f"{points:,} pts"
    
    def format_segment_budget(self, amount: Union[int, float, Decimal]) -> str:
        """Format budget amount for segment allocation"""
        return self.currency_handler.format_mobile_display(amount, 'segment_budget')
    
    def format_pricing_rate(self, rate: Union[int, float, Decimal]) -> str:
        """Format room rate for pricing game"""
        return self.currency_handler.format_mobile_display(rate, 'pricing_slider')
    
    def get_progress_color(self, current: Union[int, float, Decimal], 
                          target: Union[int, float, Decimal]) -> tuple:
        """Get color for progress indicators"""
        if target == 0:
            return (0.5, 0.5, 0.5, 1)  # Gray
        
        progress_ratio = float(current) / float(target)
        
        if progress_ratio >= 1.0:
            return (0.0, 0.8, 0.0, 1)  # Green - achieved
        elif progress_ratio >= 0.8:
            return (0.8, 0.8, 0.0, 1)  # Yellow - close
        elif progress_ratio >= 0.5:
            return (1.0, 0.5, 0.0, 1)  # Orange - halfway
        else:
            return (0.8, 0.0, 0.0, 1)  # Red - far from target

# Global formatter instances for mobile game
aed_handler = AEDCurrencyHandler()
mobile_aed_formatter = MobileAEDFormatter()

def format_aed_mobile(amount, context="general"):
    """Quick function for mobile AED formatting"""
    return mobile_aed_formatter.currency_handler.format_mobile_display(amount, context)

def aed_to_points(amount, multiplier=1.0):
    """Convert AED amount to game points"""
    return aed_handler.calculate_game_score_from_aed(amount, multiplier)

if __name__ == "__main__":
    # Test AED currency handling
    print("💰 Grand Millennium - AED Currency Handler Test")
    print("=" * 50)
    
    handler = AEDCurrencyHandler()
    mobile_formatter = MobileAEDFormatter()
    
    # Test basic formatting
    test_amounts = [1234.56, 50000, 1500000, 25.00, 0]
    
    print("📱 Mobile Formatting Tests:")
    for amount in test_amounts:
        dashboard = mobile_formatter.format_dashboard_kpi(amount, 'revenue')
        points = mobile_formatter.format_challenge_points(amount)
        pricing = mobile_formatter.format_pricing_rate(amount)
        
        print(f"  Amount: {amount}")
        print(f"    Dashboard: {dashboard}")
        print(f"    Points: {points}")
        print(f"    Pricing: {pricing}")
        print()
    
    # Test percentage changes
    print("📊 Percentage Change Tests:")
    old_revenue = 450000
    new_revenue = 520000
    change = handler.calculate_percentage_change(old_revenue, new_revenue)
    
    print(f"  Old: {handler.format_aed(old_revenue)}")
    print(f"  New: {handler.format_aed(new_revenue)}")
    print(f"  Change: {change['formatted_change']} ({change['formatted_percentage']})")
    
    # Test target formatting
    print("🎯 Target Formatting Tests:")
    current = 380000
    target = 500000
    target_info = handler.format_revenue_target(current, target)
    
    for key, value in target_info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ AED Currency Handler Ready for Mobile Game!")