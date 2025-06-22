import pandas as pd

class JobDataAnalysis:
    def __init__(self, job_data):
        self.job_data = job_data
        
    def create_dataframe(self): 
        """
        Converts the job data into a pandas DataFrame.
        """
        return pd.read_csv(self.job_data)

    def get_company_locations(self):
        """
        Extracts unique company locations from the job data.
        """
        df = self.create_dataframe()
        if 'company_location' in df.columns:
            return df['company_location'].unique().tolist()
        return []
    
    def salary_range_per_emp_type(self, country):
        """
        Calculates the salary range for each employment type.
        """
        df = self.create_dataframe()
        if 'company_location' not in df.columns or 'employment_type' not in df.columns or 'salary_usd' not in df.columns:
            print("Required columns are missing in the DataFrame.")
            print("Available columns:", df.columns)
            return {}
        else:
            filtered_df = df[df['company_location'] == country]
            salary_ranges = filtered_df.groupby('employment_type')['salary_usd'].agg(['min', 'max', 'mean'])
            return salary_ranges   

    def get_avg_exp_per_level(self, country):
        """
        Calculates the average experience required for each job level.
        """
        df = self.create_dataframe()
        if 'company_location' not in df.columns or 'experience_level' not in df.columns or 'years_experience' not in df.columns:
            print("Required columns are missing in the DataFrame.")
            print("Available columns:", df.columns)
            return {}
        else:
            filtered_df = df[df['company_location'] == country]
            avg_exp_per_level = filtered_df.groupby('experience_level')['years_experience'].mean().to_dict()
            return avg_exp_per_level
    
    def get_num_industries(self):
        """
        Counts the number of unique industries in the job data.
        """
        df = self.create_dataframe()
        if 'company_location' not in df.columns or 'industry' not in df.columns:
            print("Required columns are missing in the DataFrame.")
            print("Available columns:", df.columns)
            return {}
        else:
            number_of_industries = df.groupby('company_location')['industry' ].size().to_dict()
            return number_of_industries
    
    def get_benefit_score_range(self):
        """
        Calculates the range of benefit scores across all jobs.
        """
        df = self.create_dataframe()
        if 'company_location' not in df.columns or 'benefits_score' not in df.columns:
            print("Required columns are missing in the DataFrame.")
            print("Available columns:", df.columns)
            return {}
        else:
            benefit_scores = df.groupby('company_location')['benefits_score'].agg(['min', 'max', 'mean'])
            return benefit_scores   