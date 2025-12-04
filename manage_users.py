#!/usr/bin/env python3
"""Command-line user management tool for Touchless Lock System.

Usage:
    python manage_users.py --list
    python manage_users.py --view USER_ID
    python manage_users.py --delete USER_ID
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from db import get_all_users, delete_user, get_user_template_count, get_user_templates_info, find_user_by_name


def list_users() -> None:
    """List all registered users."""
    users = get_all_users()
    
    if not users:
        print("No users registered.")
        return
    
    print("\nRegistered Users:")
    print("-" * 60)
    print(f"{'ID':<6} | {'Name':<30} | {'Templates':<10}")
    print("-" * 60)
    
    for user_id, name in users:
        template_count = get_user_template_count(user_id)
        print(f"{user_id:<6} | {name:<30} | {template_count:<10}")
    
    print("-" * 60)
    print(f"Total: {len(users)} user(s)")


def view_user(user_id: Optional[int] = None, user_name: Optional[str] = None) -> None:
    """View detailed information about a user."""
    if user_id is None and user_name is None:
        print("Error: Must provide either --user-id or --user-name")
        sys.exit(1)
    
    if user_name is not None:
        user_id = find_user_by_name(user_name)
        if user_id is None:
            print(f"Error: User '{user_name}' not found")
            sys.exit(1)
    
    # Get user info
    users = get_all_users()
    user_info = next((uid, name) for uid, name in users if uid == user_id)
    
    if not user_info:
        print(f"Error: User ID {user_id} not found")
        sys.exit(1)
    
    _, name = user_info
    template_count = get_user_template_count(user_id)
    templates_info = get_user_templates_info(user_id)
    
    print(f"\nUser Details:")
    print("-" * 60)
    print(f"ID:       {user_id}")
    print(f"Name:     {name}")
    print(f"Templates: {template_count}")
    
    if templates_info:
        print("\nTemplate Details:")
        print("-" * 60)
        print(f"{'Template ID':<12} | {'Handedness':<12} | {'Feature Type':<20}")
        print("-" * 60)
        for template_id, handedness, feature_type in templates_info:
            print(f"{template_id:<12} | {handedness:<12} | {feature_type:<20}")
    else:
        print("\nNo templates found for this user.")
    
    print("-" * 60)


def delete_user_cmd(user_id: Optional[int] = None, user_name: Optional[str] = None, force: bool = False) -> None:
    """Delete a user and all their templates."""
    if user_id is None and user_name is None:
        print("Error: Must provide either --user-id or --user-name")
        sys.exit(1)
    
    if user_name is not None:
        user_id = find_user_by_name(user_name)
        if user_id is None:
            print(f"Error: User '{user_name}' not found")
            sys.exit(1)
    
    # Get user info for confirmation
    users = get_all_users()
    user_info = next((uid, name) for uid, name in users if uid == user_id)
    
    if not user_info:
        print(f"Error: User ID {user_id} not found")
        sys.exit(1)
    
    _, name = user_info
    template_count = get_user_template_count(user_id)
    
    # Confirm deletion
    if not force:
        print(f"\nWarning: This will delete user '{name}' (ID: {user_id}) and all {template_count} template(s).")
        response = input("Are you sure you want to delete this user? (yes/no): ").strip().lower()
        if response not in ('yes', 'y'):
            print("Deletion cancelled.")
            return
    
    # Delete user
    if delete_user(user_id):
        print(f"Successfully deleted user '{name}' (ID: {user_id}) and {template_count} template(s).")
    else:
        print(f"Error: Failed to delete user '{name}' (ID: {user_id})")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="User management tool for Touchless Lock System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_users.py --list
  python manage_users.py --view 1
  python manage_users.py --view --user-name "John Doe"
  python manage_users.py --delete 1
  python manage_users.py --delete --user-name "John Doe" --force
        """
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all registered users"
    )
    
    parser.add_argument(
        "--view", "-v",
        action="store_true",
        help="View detailed information about a user"
    )
    
    parser.add_argument(
        "--delete", "-d",
        action="store_true",
        help="Delete a user and all their templates"
    )
    
    parser.add_argument(
        "--user-id",
        type=int,
        help="User ID to view or delete"
    )
    
    parser.add_argument(
        "--user-name",
        type=str,
        help="User name to view or delete"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt when deleting"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.list, args.view, args.delete]):
        parser.print_help()
        sys.exit(1)
    
    if args.list:
        list_users()
    elif args.view:
        view_user(user_id=args.user_id, user_name=args.user_name)
    elif args.delete:
        delete_user_cmd(user_id=args.user_id, user_name=args.user_name, force=args.force)


if __name__ == "__main__":
    main()

