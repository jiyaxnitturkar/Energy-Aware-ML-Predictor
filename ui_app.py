#!/usr/bin/env python3
"""
ML Performance Predictor - Desktop UI Application
Pure Python implementation using customtkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import sys
import os
from pathlib import Path

# Import your predictor
from complete_predictor import (
    MLPerformancePredictor,
    HardwareSpec,
    BenchmarkData
)

# Configure customtkinter
ctk.set_appearance_mode("dark")  # "light" or "dark"
ctk.set_default_color_theme("blue")  # Themes: blue, dark-blue, green


class MLPredictorApp(ctk.CTk):
    """Main application class"""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("ML Performance Predictor")
        self.geometry("1200x800")
        
        # Initialize predictor
        self.predictor = None
        self.is_model_trained = False
        
        # Create UI
        self.create_widgets()
        
        # Initialize predictor in background
        self.after(100, self.initialize_predictor_async)
        
    def create_widgets(self):
        """Create all UI widgets"""
        
        # Header
        header_frame = ctk.CTkFrame(self, corner_radius=10)
        header_frame.pack(fill="x", padx=20, pady=10)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🤖 ML Performance Predictor",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=15)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Compare local vs cloud training performance and get AI-powered recommendations",
            font=ctk.CTkFont(size=14)
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Status bar
        self.status_label = ctk.CTkLabel(
            header_frame,
            text="⏳ Initializing models...",
            font=ctk.CTkFont(size=12),
            text_color="orange"
        )
        self.status_label.pack(pady=(0, 10))
        
        # Main container with tabs
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Add tabs
        self.tabview.add("Predict Performance")
        self.tabview.add("Hardware Setup")
        self.tabview.add("Results")
        
        # Create tab contents
        self.create_predict_tab()
        self.create_hardware_tab()
        self.create_results_tab()
        
    def create_predict_tab(self):
        """Create prediction tab content"""
        tab = self.tabview.tab("Predict Performance")
        
        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Preset Models Section
        preset_frame = ctk.CTkFrame(scroll_frame)
        preset_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            preset_frame,
            text="Quick Presets",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        presets_button_frame = ctk.CTkFrame(preset_frame)
        presets_button_frame.pack(fill="x", padx=10, pady=10)
        
        presets = [
            ("ResNet-50", "25600000", "4100000000", "ImageNet", "1281167"),
            ("BERT-Large", "340000000", "130000000000", "Wikipedia", "6000000"),
            ("GPT-2", "1500000000", "3100000000000", "OpenWebText", "40000000"),
            ("ViT-Large", "307000000", "61000000000", "ImageNet", "1281167")
        ]
        
        for i, (name, params, flops, dataset, ds_size) in enumerate(presets):
            btn = ctk.CTkButton(
                presets_button_frame,
                text=name,
                width=140,
                command=lambda n=name, p=params, f=flops, d=dataset, s=ds_size: 
                    self.load_preset(n, p, f, d, s)
            )
            btn.grid(row=0, column=i, padx=5, pady=5)
        
        # Model Configuration Section
        config_frame = ctk.CTkFrame(scroll_frame)
        config_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            config_frame,
            text="Model Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        # Form fields
        form_frame = ctk.CTkFrame(config_frame)
        form_frame.pack(fill="x", padx=10, pady=10)
        
        # Model Name
        self.model_name_var = ctk.StringVar(value="ResNet-50")
        self.create_form_field(form_frame, "Model Name:", self.model_name_var, 0)
        
        # Dataset
        self.dataset_var = ctk.StringVar(value="ImageNet")
        self.create_form_field(form_frame, "Dataset:", self.dataset_var, 1)
        
        # Model Parameters
        self.model_params_var = ctk.StringVar(value="25600000")
        self.create_form_field(form_frame, "Model Parameters:", self.model_params_var, 2)
        
        # FLOPs
        self.flops_var = ctk.StringVar(value="4100000000")
        self.create_form_field(form_frame, "FLOPs:", self.flops_var, 3)
        
        # Batch Size
        self.batch_size_var = ctk.StringVar(value="32")
        self.create_form_field(form_frame, "Batch Size:", self.batch_size_var, 4)
        
        # Epochs
        self.epochs_var = ctk.StringVar(value="90")
        self.create_form_field(form_frame, "Epochs:", self.epochs_var, 5)
        
        # Dataset Size
        self.dataset_size_var = ctk.StringVar(value="1281167")
        self.create_form_field(form_frame, "Dataset Size:", self.dataset_size_var, 6)
        
        # Framework
        self.framework_var = ctk.StringVar(value="PyTorch")
        self.create_dropdown_field(
            form_frame, "Framework:", self.framework_var, 
            ["PyTorch", "TensorFlow"], 7
        )
        
        # Precision
        self.precision_var = ctk.StringVar(value="FP32")
        self.create_dropdown_field(
            form_frame, "Precision:", self.precision_var,
            ["FP32", "FP16"], 8
        )
        
        # Predict Button
        self.predict_button = ctk.CTkButton(
            scroll_frame,
            text="🔮 Generate Prediction",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            command=self.generate_prediction
        )
        self.predict_button.pack(fill="x", padx=10, pady=20)
        
    def create_hardware_tab(self):
        """Create hardware setup tab content"""
        tab = self.tabview.tab("Hardware Setup")
        
        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Info box
        info_frame = ctk.CTkFrame(scroll_frame, fg_color=("lightblue", "#1f538d"))
        info_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            info_frame,
            text="ℹ️ Configure your local hardware specifications for accurate predictions",
            font=ctk.CTkFont(size=12),
            wraplength=700
        ).pack(padx=15, pady=15)
        
        # Hardware Configuration
        hw_frame = ctk.CTkFrame(scroll_frame)
        hw_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            hw_frame,
            text="Local Hardware Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        form_frame = ctk.CTkFrame(hw_frame)
        form_frame.pack(fill="x", padx=10, pady=10)
        
        # GPU Type
        self.gpu_type_var = ctk.StringVar(value="RTX 3090")
        self.create_dropdown_field(
            form_frame, "GPU Type:", self.gpu_type_var,
            ["RTX 3060", "RTX 3070", "RTX 3080", "RTX 3090", 
             "RTX 4070", "RTX 4080", "RTX 4090", 
             "A100", "V100", "H100"], 0
        )
        
        # GPU Memory
        self.gpu_memory_var = ctk.StringVar(value="24")
        self.create_form_field(form_frame, "GPU Memory (GB):", self.gpu_memory_var, 1)
        
        # CPU Type
        self.cpu_type_var = ctk.StringVar(value="Intel Core i9")
        self.create_form_field(form_frame, "CPU Type:", self.cpu_type_var, 2)
        
        # RAM
        self.ram_var = ctk.StringVar(value="32")
        self.create_form_field(form_frame, "RAM (GB):", self.ram_var, 3)
        
        # Power Cost
        self.power_cost_var = ctk.StringVar(value="0.12")
        self.create_form_field(form_frame, "Power Cost ($/kWh):", self.power_cost_var, 4)
        
        # Save button
        save_button = ctk.CTkButton(
            scroll_frame,
            text="💾 Save Hardware Configuration",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self.save_hardware_config
        )
        save_button.pack(fill="x", padx=10, pady=20)
        
    def create_results_tab(self):
        """Create results tab content"""
        tab = self.tabview.tab("Results")
        
        # Scrollable frame
        self.results_frame = ctk.CTkScrollableFrame(tab)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Placeholder
        self.results_placeholder = ctk.CTkLabel(
            self.results_frame,
            text="📊 No predictions yet. Go to 'Predict Performance' to generate a prediction.",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.results_placeholder.pack(expand=True, pady=100)
        
    def create_form_field(self, parent, label_text, variable, row):
        """Create a form field with label and entry"""
        label = ctk.CTkLabel(parent, text=label_text, font=ctk.CTkFont(size=12))
        label.grid(row=row, column=0, sticky="w", padx=10, pady=8)
        
        entry = ctk.CTkEntry(parent, textvariable=variable, width=200)
        entry.grid(row=row, column=1, sticky="ew", padx=10, pady=8)
        
        parent.grid_columnconfigure(1, weight=1)
        
    def create_dropdown_field(self, parent, label_text, variable, values, row):
        """Create a dropdown field with label"""
        label = ctk.CTkLabel(parent, text=label_text, font=ctk.CTkFont(size=12))
        label.grid(row=row, column=0, sticky="w", padx=10, pady=8)
        
        dropdown = ctk.CTkOptionMenu(parent, variable=variable, values=values, width=200)
        dropdown.grid(row=row, column=1, sticky="ew", padx=10, pady=8)
        
        parent.grid_columnconfigure(1, weight=1)
        
    def load_preset(self, name, params, flops, dataset, ds_size):
        """Load preset model values"""
        self.model_name_var.set(name)
        self.model_params_var.set(params)
        self.flops_var.set(flops)
        self.dataset_var.set(dataset)
        self.dataset_size_var.set(ds_size)
        
        messagebox.showinfo("Preset Loaded", f"Loaded {name} configuration")
        
    def initialize_predictor_async(self):
        """Initialize predictor in background thread"""
        def init_task():
            try:
                self.status_label.configure(text="⏳ Loading benchmark data...")
                self.predictor = MLPerformancePredictor()
                
                # Collect data
                csv_file = r'C:\Users\Krithika\ml_performance_predictor\dataset.csv'
                if os.path.exists(csv_file):
                    benchmarks = self.predictor.collect_data(scrape_web=True, csv_files=[csv_file])
                else:
                    benchmarks = self.predictor.collect_data(scrape_web=True)
                
                self.status_label.configure(text="🧠 Training ML models...")
                
                # Train models
                self.predictor.train_models(benchmarks, model_type="xgboost")
                self.is_model_trained = True
                
                self.status_label.configure(
                    text="✅ Models trained successfully! Ready to predict.",
                    text_color="green"
                )
                self.predict_button.configure(state="normal")
                
            except Exception as e:
                self.status_label.configure(
                    text=f"❌ Initialization failed: {str(e)}",
                    text_color="red"
                )
                messagebox.showerror("Error", f"Failed to initialize predictor:\n{str(e)}")
        
        thread = threading.Thread(target=init_task, daemon=True)
        thread.start()
        
    def save_hardware_config(self):
        """Save hardware configuration"""
        messagebox.showinfo(
            "Configuration Saved",
            "Hardware configuration saved successfully!\n\n"
            f"GPU: {self.gpu_type_var.get()} ({self.gpu_memory_var.get()}GB)\n"
            f"CPU: {self.cpu_type_var.get()}\n"
            f"RAM: {self.ram_var.get()}GB\n"
            f"Power Cost: ${self.power_cost_var.get()}/kWh"
        )
        
    def generate_prediction(self):
        """Generate performance prediction"""
        if not self.is_model_trained:
            messagebox.showwarning(
                "Not Ready",
                "Models are still training. Please wait a moment."
            )
            return
        
        try:
            # Get form values
            model_name = self.model_name_var.get()
            model_params = int(self.model_params_var.get())
            flops = float(self.flops_var.get())
            batch_size = int(self.batch_size_var.get())
            epochs = int(self.epochs_var.get())
            dataset = self.dataset_var.get()
            dataset_size = int(self.dataset_size_var.get())
            framework = self.framework_var.get()
            precision = self.precision_var.get()
            
            # Hardware specs
            local_hardware = HardwareSpec(
                gpu_type=self.gpu_type_var.get(),
                gpu_memory_gb=float(self.gpu_memory_var.get()),
                cpu_type=self.cpu_type_var.get(),
                ram_gb=float(self.ram_var.get()),
                storage_gb=1000,
                power_cost_per_kwh=float(self.power_cost_var.get())
            )
            
            # Show progress
            self.status_label.configure(text="🔄 Generating prediction...")
            self.predict_button.configure(state="disabled")
            
            def predict_task():
                try:
                    # Get recommendation
                    recommendation = self.predictor.recommend_training_approach(
                        model_name=model_name,
                        dataset=dataset,
                        batch_size=batch_size,
                        epochs=epochs,
                        dataset_size=dataset_size,
                        model_params=model_params,
                        flops=flops,
                        framework=framework,
                        precision=precision,
                        local_hardware=local_hardware
                    )
                    
                    # Display results
                    self.after(0, lambda: self.display_results(recommendation, model_name))
                    self.after(0, lambda: self.status_label.configure(
                        text="✅ Prediction complete!",
                        text_color="green"
                    ))
                    
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Error", f"Prediction failed:\n{str(e)}"))
                    self.after(0, lambda: self.status_label.configure(
                        text="❌ Prediction failed",
                        text_color="red"
                    ))
                finally:
                    self.after(0, lambda: self.predict_button.configure(state="normal"))
            
            thread = threading.Thread(target=predict_task, daemon=True)
            thread.start()
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your input values:\n{str(e)}")
            self.predict_button.configure(state="normal")
            
    def display_results(self, recommendation, model_name):
        """Display prediction results"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Switch to results tab
        self.tabview.set("Results")
        
        # Recommendation Header
        rec_color = "green" if recommendation.recommendation == "local" else "blue"
        rec_emoji = "🏠" if recommendation.recommendation == "local" else "☁️"
        rec_text = "TRAIN LOCALLY" if recommendation.recommendation == "local" else "TRAIN ON CLOUD"
        
        header_frame = ctk.CTkFrame(self.results_frame, fg_color=rec_color, corner_radius=10)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            header_frame,
            text=f"{rec_emoji} {rec_text}",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white"
        ).pack(pady=10)
        
        ctk.CTkLabel(
            header_frame,
            text=f"Recommended for {model_name}",
            font=ctk.CTkFont(size=14),
            text_color="white"
        ).pack(pady=(0, 5))
        
        # Confidence and Feasibility
        metrics_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        metrics_frame.pack(fill="x", padx=20, pady=10)
        
        conf_frame = ctk.CTkFrame(metrics_frame, fg_color="white", corner_radius=8)
        conf_frame.pack(side="left", expand=True, fill="x", padx=5)
        
        ctk.CTkLabel(
            conf_frame,
            text=f"{recommendation.confidence*100:.0f}%",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=rec_color
        ).pack(pady=5)
        
        ctk.CTkLabel(
            conf_frame,
            text="Confidence",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=(0, 5))
        
        feas_frame = ctk.CTkFrame(metrics_frame, fg_color="white", corner_radius=8)
        feas_frame.pack(side="left", expand=True, fill="x", padx=5)
        
        ctk.CTkLabel(
            feas_frame,
            text=f"{recommendation.feasibility_score*100:.0f}%",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=rec_color
        ).pack(pady=5)
        
        ctk.CTkLabel(
            feas_frame,
            text="Feasibility Score",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=(0, 5))
        
        # Performance Comparison
        comp_frame = ctk.CTkFrame(self.results_frame)
        comp_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            comp_frame,
            text="📈 Performance Comparison",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=10, pady=10)
        
        comparison_grid = ctk.CTkFrame(comp_frame)
        comparison_grid.pack(fill="x", padx=10, pady=10)
        
        # Time
        self.create_comparison_card(
            comparison_grid,
            "⏱️ Training Time",
            f"{recommendation.local_prediction['training_time']:.1f}h",
            f"{recommendation.cloud_prediction['training_time']:.1f}h",
            f"{recommendation.time_difference:+.1f}h",
            0
        )
        
        # Energy
        self.create_comparison_card(
            comparison_grid,
            "⚡ Energy Usage",
            f"{recommendation.local_prediction['energy_usage']:.1f} kWh",
            f"{recommendation.cloud_prediction['energy_usage']:.1f} kWh",
            f"{recommendation.local_prediction['energy_usage']-recommendation.cloud_prediction['energy_usage']:+.1f} kWh",
            1
        )
        
        # Cost
        self.create_comparison_card(
            comparison_grid,
            "💰 Total Cost",
            f"${recommendation.local_prediction['cost_usd']:.2f}",
            f"${recommendation.cloud_prediction['cost_usd']:.2f}",
            f"${recommendation.cost_savings:+.2f}",
            2
        )
        
        # Cost Savings Highlight
        if recommendation.cost_savings > 0:
            savings_text = f"💰 POTENTIAL SAVINGS: ${recommendation.cost_savings:.2f} with local training"
            savings_color = "green"
        else:
            savings_text = f"💸 ADDITIONAL COST: ${abs(recommendation.cost_savings):.2f} for local training"
            savings_color = "orange"
        
        savings_frame = ctk.CTkFrame(self.results_frame, fg_color=savings_color)
        savings_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            savings_frame,
            text=savings_text,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        ).pack(pady=10)
        
        # Detailed Analysis
        analysis_frame = ctk.CTkFrame(self.results_frame)
        analysis_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            analysis_frame,
            text="🔍 Detailed Analysis",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=10)
        
        # Reasoning text
        reasoning_text = scrolledtext.ScrolledText(
            analysis_frame,
            wrap=tk.WORD,
            height=12,
            font=("Consolas", 11),
            bg="#2b2b2b",
            fg="#ffffff",
            insertbackground="white"
        )
        reasoning_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        for reason in recommendation.reasoning:
            reasoning_text.insert(tk.END, f"• {reason}\n")
        
        reasoning_text.configure(state="disabled")
        
    def create_comparison_card(self, parent, title, local_val, cloud_val, diff_val, col):
        """Create a comparison card"""
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")
        
        parent.grid_columnconfigure(col, weight=1)
        
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Local
        local_frame = ctk.CTkFrame(card, fg_color="transparent")
        local_frame.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(
            local_frame,
            text="Local:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left")
        
        ctk.CTkLabel(
            local_frame,
            text=local_val,
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="right")
        
        # Cloud
        cloud_frame = ctk.CTkFrame(card, fg_color="transparent")
        cloud_frame.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(
            cloud_frame,
            text="Cloud:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left")
        
        ctk.CTkLabel(
            cloud_frame,
            text=cloud_val,
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="right")
        
        # Difference
        diff_color = "green" if "-" in diff_val or "Save" in diff_val else "red"
        
        diff_label = ctk.CTkLabel(
            card,
            text=diff_val,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=diff_color
        )
        diff_label.pack(pady=(5, 10))


def main():
    """Main entry point"""
    app = MLPredictorApp()
    app.mainloop()


if __name__ == "__main__":
    main()