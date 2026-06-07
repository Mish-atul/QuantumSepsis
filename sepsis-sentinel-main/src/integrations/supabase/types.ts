export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.4"
  }
  public: {
    Tables: {
      hospitals: {
        Row: {
          created_at: string
          id: string
          name: string
          tier: string
          total_icu_beds: number
        }
        Insert: {
          created_at?: string
          id?: string
          name: string
          tier?: string
          total_icu_beds?: number
        }
        Update: {
          created_at?: string
          id?: string
          name?: string
          tier?: string
          total_icu_beds?: number
        }
        Relationships: []
      }
      labs: {
        Row: {
          bilirubin: number | null
          creatinine: number | null
          hospital_id: string | null
          id: string
          lactate: number | null
          patient_id: string
          platelets: number | null
          procalcitonin: number | null
          timestamp: string
          wbc: number | null
        }
        Insert: {
          bilirubin?: number | null
          creatinine?: number | null
          hospital_id?: string | null
          id?: string
          lactate?: number | null
          patient_id: string
          platelets?: number | null
          procalcitonin?: number | null
          timestamp?: string
          wbc?: number | null
        }
        Update: {
          bilirubin?: number | null
          creatinine?: number | null
          hospital_id?: string | null
          id?: string
          lactate?: number | null
          patient_id?: string
          platelets?: number | null
          procalcitonin?: number | null
          timestamp?: string
          wbc?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "labs_hospital_id_fkey"
            columns: ["hospital_id"]
            isOneToOne: false
            referencedRelation: "hospitals"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "labs_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      patients: {
        Row: {
          admission_time: string
          bed_number: string
          created_at: string
          hospital_id: string | null
          id: string
          mrn: string
          name: string
          status: string
        }
        Insert: {
          admission_time?: string
          bed_number: string
          created_at?: string
          hospital_id?: string | null
          id?: string
          mrn: string
          name: string
          status?: string
        }
        Update: {
          admission_time?: string
          bed_number?: string
          created_at?: string
          hospital_id?: string | null
          id?: string
          mrn?: string
          name?: string
          status?: string
        }
        Relationships: [
          {
            foreignKeyName: "patients_hospital_id_fkey"
            columns: ["hospital_id"]
            isOneToOne: false
            referencedRelation: "hospitals"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          avatar_url: string | null
          created_at: string
          department: string | null
          employee_id: string | null
          full_name: string | null
          hospital_id: string | null
          id: string
          role: string
          updated_at: string
          user_id: string
        }
        Insert: {
          avatar_url?: string | null
          created_at?: string
          department?: string | null
          employee_id?: string | null
          full_name?: string | null
          hospital_id?: string | null
          id?: string
          role?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          avatar_url?: string | null
          created_at?: string
          department?: string | null
          employee_id?: string | null
          full_name?: string | null
          hospital_id?: string | null
          id?: string
          role?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "profiles_hospital_id_fkey"
            columns: ["hospital_id"]
            isOneToOne: false
            referencedRelation: "hospitals"
            referencedColumns: ["id"]
          },
        ]
      }
      risk_assessments: {
        Row: {
          confidence_interval_lower: number
          confidence_interval_upper: number
          hospital_id: string | null
          id: string
          patient_id: string
          quantum_risk_score: number
          tier: string
          timestamp: string
        }
        Insert: {
          confidence_interval_lower: number
          confidence_interval_upper: number
          hospital_id?: string | null
          id?: string
          patient_id: string
          quantum_risk_score: number
          tier: string
          timestamp?: string
        }
        Update: {
          confidence_interval_lower?: number
          confidence_interval_upper?: number
          hospital_id?: string | null
          id?: string
          patient_id?: string
          quantum_risk_score?: number
          tier?: string
          timestamp?: string
        }
        Relationships: [
          {
            foreignKeyName: "risk_assessments_hospital_id_fkey"
            columns: ["hospital_id"]
            isOneToOne: false
            referencedRelation: "hospitals"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "risk_assessments_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      tripwire_alerts: {
        Row: {
          hospital_id: string | null
          id: string
          is_active: boolean
          metric: string
          patient_id: string
          threshold_breached: string
          timestamp: string
          value: number
        }
        Insert: {
          hospital_id?: string | null
          id?: string
          is_active?: boolean
          metric: string
          patient_id: string
          threshold_breached: string
          timestamp?: string
          value: number
        }
        Update: {
          hospital_id?: string | null
          id?: string
          is_active?: boolean
          metric?: string
          patient_id?: string
          threshold_breached?: string
          timestamp?: string
          value?: number
        }
        Relationships: [
          {
            foreignKeyName: "tripwire_alerts_hospital_id_fkey"
            columns: ["hospital_id"]
            isOneToOne: false
            referencedRelation: "hospitals"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tripwire_alerts_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      vitals: {
        Row: {
          blood_pressure_dia: number | null
          blood_pressure_sys: number | null
          heart_rate: number | null
          hospital_id: string | null
          id: string
          is_manual_entry: boolean
          map: number | null
          mental_status: string
          patient_id: string
          respiratory_rate: number | null
          spo2: number | null
          temperature: number | null
          timestamp: string
        }
        Insert: {
          blood_pressure_dia?: number | null
          blood_pressure_sys?: number | null
          heart_rate?: number | null
          hospital_id?: string | null
          id?: string
          is_manual_entry?: boolean
          map?: number | null
          mental_status?: string
          patient_id: string
          respiratory_rate?: number | null
          spo2?: number | null
          temperature?: number | null
          timestamp?: string
        }
        Update: {
          blood_pressure_dia?: number | null
          blood_pressure_sys?: number | null
          heart_rate?: number | null
          hospital_id?: string | null
          id?: string
          is_manual_entry?: boolean
          map?: number | null
          mental_status?: string
          patient_id?: string
          respiratory_rate?: number | null
          spo2?: number | null
          temperature?: number | null
          timestamp?: string
        }
        Relationships: [
          {
            foreignKeyName: "vitals_hospital_id_fkey"
            columns: ["hospital_id"]
            isOneToOne: false
            referencedRelation: "hospitals"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "vitals_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      get_user_hospital_id: { Args: { _user_id: string }; Returns: string }
      get_user_role: { Args: { _user_id: string }; Returns: string }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
