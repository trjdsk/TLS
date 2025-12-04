#!/usr/bin/env python3
"""GUI user management tool for Touchless Lock System.

A Windows-friendly graphical interface for managing registered users.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List, Tuple
import sys

from db import get_all_users, delete_user, get_user_template_count, get_user_templates_info


class UserManagementGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Touchless Lock System - User Management")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables
        self.users_list: List[Tuple[int, str]] = []
        self.selected_user_id: Optional[int] = None
        
        self._create_widgets()
        self.refresh_users_list()
    
    def _create_widgets(self) -> None:
        """Create and layout all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Registered Users", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Left panel - Users list
        list_frame = ttk.LabelFrame(main_frame, text="Users", padding="10")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Listbox with scrollbar
        list_scrollbar = ttk.Scrollbar(list_frame)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.users_listbox = tk.Listbox(list_frame, height=20, width=40, yscrollcommand=list_scrollbar.set)
        self.users_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.users_listbox.bind('<<ListboxSelect>>', self.on_user_select)
        list_scrollbar.config(command=self.users_listbox.yview)
        
        # Buttons frame
        buttons_frame = ttk.Frame(list_frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        refresh_btn = ttk.Button(buttons_frame, text="Refresh", command=self.refresh_users_list)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        delete_btn = ttk.Button(buttons_frame, text="Delete Selected", command=self.delete_selected_user)
        delete_btn.pack(side=tk.LEFT)
        
        # Right panel - User details
        details_frame = ttk.LabelFrame(main_frame, text="User Details", padding="10")
        details_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Details text widget with scrollbar
        details_scrollbar = ttk.Scrollbar(details_frame)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.details_text = tk.Text(details_frame, height=20, width=40, 
                                   yscrollcommand=details_scrollbar.set, wrap=tk.WORD)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.config(command=self.details_text.yview)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def refresh_users_list(self) -> None:
        """Refresh the list of users from the database."""
        try:
            self.users_list = get_all_users()
            self.users_listbox.delete(0, tk.END)
            
            for user_id, name in self.users_list:
                template_count = get_user_template_count(user_id)
                display_text = f"ID: {user_id} | {name} ({template_count} template(s))"
                self.users_listbox.insert(tk.END, display_text)
            
            self.status_label.config(text=f"Loaded {len(self.users_list)} user(s)")
            
            # Clear details if no users
            if not self.users_list:
                self.details_text.delete(1.0, tk.END)
                self.details_text.insert(1.0, "No users registered.")
                self.selected_user_id = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load users: {e}")
            self.status_label.config(text="Error loading users")
    
    def on_user_select(self, event: tk.Event) -> None:
        """Handle user selection from listbox."""
        selection = self.users_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if 0 <= index < len(self.users_list):
            user_id, name = self.users_list[index]
            self.selected_user_id = user_id
            self.show_user_details(user_id, name)
    
    def show_user_details(self, user_id: int, name: str) -> None:
        """Display detailed information about the selected user."""
        try:
            template_count = get_user_template_count(user_id)
            templates_info = get_user_templates_info(user_id)
            
            self.details_text.delete(1.0, tk.END)
            
            details = f"User Details\n"
            details += "=" * 50 + "\n\n"
            details += f"ID:       {user_id}\n"
            details += f"Name:     {name}\n"
            details += f"Templates: {template_count}\n\n"
            
            if templates_info:
                details += "Template Details:\n"
                details += "-" * 50 + "\n"
                details += f"{'Template ID':<15} | {'Handedness':<12} | {'Feature Type'}\n"
                details += "-" * 50 + "\n"
                for template_id, handedness, feature_type in templates_info:
                    details += f"{template_id:<15} | {handedness:<12} | {feature_type}\n"
            else:
                details += "\nNo templates found for this user.\n"
            
            self.details_text.insert(1.0, details)
            self.status_label.config(text=f"Viewing user: {name} (ID: {user_id})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load user details: {e}")
            self.status_label.config(text="Error loading user details")
    
    def delete_selected_user(self) -> None:
        """Delete the currently selected user."""
        if self.selected_user_id is None:
            messagebox.showwarning("No Selection", "Please select a user to delete.")
            return
        
        # Find user name
        user_info = next((uid, name) for uid, name in self.users_list if uid == self.selected_user_id)
        if not user_info:
            messagebox.showerror("Error", "Selected user not found.")
            return
        
        _, name = user_info
        template_count = get_user_template_count(self.selected_user_id)
        
        # Confirm deletion
        response = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete user '{name}' (ID: {self.selected_user_id})?\n\n"
            f"This will also delete {template_count} template(s) associated with this user.\n\n"
            "This action cannot be undone.",
            icon='warning'
        )
        
        if not response:
            return
        
        # Delete user
        try:
            if delete_user(self.selected_user_id):
                messagebox.showinfo("Success", f"Successfully deleted user '{name}' (ID: {self.selected_user_id})")
                self.selected_user_id = None
                self.refresh_users_list()
                self.details_text.delete(1.0, tk.END)
                self.status_label.config(text=f"Deleted user: {name}")
            else:
                messagebox.showerror("Error", f"Failed to delete user '{name}' (ID: {self.selected_user_id})")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting user: {e}")


def main() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    app = UserManagementGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

