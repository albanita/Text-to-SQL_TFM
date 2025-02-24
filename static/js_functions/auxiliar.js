document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("generate_button").addEventListener("click", function () {
        let nl_query = document.getElementById("nl_query").value;

        fetch('/generate_sql', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nl_query: nl_query })
        })
        .then(response => response.json())
        .then(data => {
            // Fill the SQL textarea
            document.getElementById("sql_generated_query").value = data.sql_query;

            // Populate the table inside the modal
            let tableHead = document.getElementById("table_head");
            let tableBody = document.getElementById("table_body");

            // Clear previous data
            tableHead.innerHTML = "";
            tableBody.innerHTML = "";

            // Add column headers
            let headerRow = document.createElement("tr");
            data.columns_list.forEach(col => {
                let th = document.createElement("th");
                th.textContent = col;
                headerRow.appendChild(th);
            });
            tableHead.appendChild(headerRow);

            // Add rows
            data.values_list.forEach(row => {
                let tr = document.createElement("tr");
                row.forEach(value => {
                    let td = document.createElement("td");
                    td.textContent = value;
                    tr.appendChild(td);
                });
                tableBody.appendChild(tr);
            });

            // Show modal
            let myModal = new bootstrap.Modal(document.getElementById('sql_table_result'));
            myModal.show();
        })
        .catch(error => console.error("Error:", error));
    });
});